"""Crash-safe cached training state and resume compatibility checks.

The ordinary ``*.safetensors`` outputs remain portable adapter checkpoints.
This module stores the larger, trainer-specific state (optimizer, RNG, and
progress) under the dataset cache so an interrupted run can continue exactly.

State publication is transactional: a uniquely named payload is fully written
and fsynced before an atomic manifest swap makes it visible.  A crash during a
save therefore leaves either the previous complete generation or the new one,
never a half-written state advertised as resumable.
"""
from __future__ import annotations

from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import PackageNotFoundError, version
import json
import os
from pathlib import Path
import random
import re
import sys
import time
from typing import Any, Callable
from uuid import uuid4

import numpy as np
import torch
import xxhash

from .config import Config


STATE_FORMAT_VERSION = 1
# Bump this whenever training math or state interpretation changes in a way
# that cannot safely continue an existing optimizer trajectory.
TRAINING_SEMANTICS_VERSION = 1

_RUNTIME_DISTRIBUTIONS = (
    "numpy",
    "torch",
    "triton",
    "transformers",
    "tokenizers",
    "liger-kernel",
    "transformer-engine",
    "transformer-engine-cu13",
    "transformer-engine-torch",
    "nvidia-cublas",
    "nvidia-cuda-runtime",
    "nvidia-cudnn-cu13",
    "nvidia-cudnn-frontend",
    "prodigy-plus-schedule-free",
    "lycoris-lora",
)


class TrainingStateError(RuntimeError):
    """Cached state is corrupt, incompatible, or cannot be restored safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def compatibility_digest(compatibility: dict[str, Any]) -> str:
    return sha256(_canonical_json(compatibility).encode("utf-8")).hexdigest()


def _file_sha256(path: Path) -> str:
    hasher = sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1 << 20):
            hasher.update(chunk)
    return hasher.hexdigest()


def runtime_versions() -> dict[str, str]:
    versions = {"python": ".".join(map(str, sys.version_info[:3]))}
    for distribution in _RUNTIME_DISTRIBUTIONS:
        try:
            versions[distribution] = version(distribution)
        except PackageNotFoundError:
            versions[distribution] = "missing"
    return versions


def build_compatibility(
    cfg: Config,
    *,
    dit_fp: str,
    vae_fp: str,
    te_fp: str,
    data_cache_fp: str,
) -> dict[str, Any]:
    """Return the state-restoration contract for a training run.

    Operational knobs (worker count, save/sample cadence, and target epoch)
    are intentionally excluded: changing them does not alter the shape or
    interpretation of an already accumulated optimizer state.  Everything
    that affects model topology, cached inputs, or training math is included.
    """
    return {
        "training_semantics": TRAINING_SEMANTICS_VERSION,
        "models": {"dit": dit_fp, "vae": vae_fp, "text_encoder": te_fp},
        "data_cache": data_cache_fp,
        "train": {
            "resolution": cfg.train.resolution,
            "batch_size": cfg.train.batch_size,
            "precision": cfg.train.precision,
            "seed": cfg.train.seed,
            "gradient_checkpointing": cfg.train.gradient_checkpointing,
            "cuda_graphs": cfg.train.cuda_graphs,
            "cuda_graph_warmup_steps": cfg.train.cuda_graph_warmup_steps,
            "compile_mode": cfg.train.compile_mode,
        },
        "optimizer": asdict(cfg.optim),
        "lokr": {
            **asdict(cfg.lokr),
            "network_dim": 128,
            "network_alpha": 128.0,
        },
        "runtime": runtime_versions(),
    }


def _hash_field(hasher, value: str | bytes) -> None:
    raw = value.encode("utf-8") if isinstance(value, str) else value
    hasher.update(len(raw).to_bytes(8, "little"))
    hasher.update(raw)


def validate_and_fingerprint_cached_data(
    samples,
    *,
    cache,
    dataset_root: str | Path,
    vae_fp: str,
    te_fp: str,
):
    """Validate cached rows, enrich sample buckets, and hash exact inputs.

    The digest includes sample order/repeats, captions, latent bytes, prompt
    embedding bytes, masks, and their model/source fingerprints.  A cache
    rewrite or dataset edit therefore invalidates optimizer state even when
    paths and tensor shapes happen to stay the same.
    """
    root = Path(dataset_root)
    sources: dict[str, tuple[Any, Any]] = {}
    texts: dict[str, Any] = {}
    enriched = []

    for sample in samples:
        if sample.src_path not in sources:
            source_file = root / sample.src_path
            crop = cache.get_crop(sample.src_path, source_file=str(source_file))
            if crop is None:
                raise RuntimeError(
                    f"missing or stale cached crop for {sample.src_path}; run precompute first"
                )
            latent = cache.get_latent(
                sample.src_path,
                vae_fp=vae_fp,
                source_file=str(source_file),
            )
            if latent is None:
                raise RuntimeError(
                    f"missing or stale cached latent for {sample.src_path}; run precompute first"
                )
            if latent.bucket_idx != crop.bucket_idx:
                raise RuntimeError(
                    f"cache bucket mismatch for {sample.src_path}: "
                    f"crop={crop.bucket_idx}, latent={latent.bucket_idx}; run precompute first"
                )
            sources[sample.src_path] = (crop, latent)

        if sample.caption not in texts:
            text = cache.get_text(sample.caption, te_fp=te_fp)
            if text is None:
                raise RuntimeError(
                    f"missing or stale cached text embed for caption[{sample.src_path}]; "
                    "run precompute first"
                )
            texts[sample.caption] = text

        crop, _ = sources[sample.src_path]
        enriched.append(
            sample.__class__(
                src_path=sample.src_path,
                bucket_idx=crop.bucket_idx,
                caption=sample.caption,
            )
        )

    hasher = xxhash.xxh3_128()
    _hash_field(hasher, "anima-training-data-v1")

    # Exact ordered training sequence, including repeats.
    for sample in enriched:
        _hash_field(hasher, sample.src_path)
        _hash_field(hasher, str(sample.bucket_idx))
        _hash_field(hasher, sample.caption)

    # Exact cached tensor payloads, each included once.
    for src_path in sorted(sources):
        crop, latent = sources[src_path]
        for value in (
            src_path,
            crop.src_xxhash,
            str(crop.bucket_idx),
            str(crop.bucket_w),
            str(crop.bucket_h),
            latent.src_xxhash,
            latent.vae_fp,
            str(latent.bucket_idx),
            latent.dtype,
            repr(latent.shape),
        ):
            _hash_field(hasher, value)
        _hash_field(hasher, latent.data)

    for caption in sorted(texts):
        text = texts[caption]
        for value in (
            caption,
            text.caption_xxhash,
            text.te_fp,
            text.dtype,
            repr(text.shape),
            repr(text.mask_shape),
        ):
            _hash_field(hasher, value)
        _hash_field(hasher, text.data)
        _hash_field(hasher, text.mask_data)

    return enriched, hasher.hexdigest()


def capture_rng_state() -> dict[str, Any]:
    np_name, np_keys, np_pos, np_has_gauss, np_cached_gaussian = np.random.get_state()
    cuda_states = torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else []
    return {
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone().cpu() for state in cuda_states],
        "python": random.getstate(),
        "numpy": {
            "name": np_name,
            "keys": torch.from_numpy(np_keys.copy()),
            "pos": int(np_pos),
            "has_gauss": int(np_has_gauss),
            "cached_gaussian": float(np_cached_gaussian),
        },
    }


def restore_rng_state(state: dict[str, Any]) -> None:
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state.get("torch_cuda", [])
    if cuda_states:
        if not torch.cuda.is_available():
            raise TrainingStateError("cached state contains CUDA RNG state but CUDA is unavailable")
        if len(cuda_states) != torch.cuda.device_count():
            raise TrainingStateError(
                "CUDA device count changed: cached "
                f"{len(cuda_states)}, current {torch.cuda.device_count()}"
            )
        torch.cuda.set_rng_state_all([item.cpu() for item in cuda_states])
    random.setstate(state["python"])
    np_state = state["numpy"]
    np.random.set_state(
        (
            np_state["name"],
            np_state["keys"].cpu().numpy().astype(np.uint32, copy=True),
            int(np_state["pos"]),
            int(np_state["has_gauss"]),
            float(np_state["cached_gaussian"]),
        )
    )


def _to_cpu_snapshot(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _to_cpu_snapshot(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu_snapshot(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu_snapshot(item) for item in value)
    return value


def _state_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.detach().to(device=device, copy=True)
    if isinstance(value, dict):
        return {key: _state_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_state_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_state_to_device(item, device) for item in value)
    return value


def restore_optimizer_state(optimizer, state_dict: dict[str, Any]) -> None:
    """Restore optimizer state without PyTorch's implicit dtype coercion.

    ``Optimizer.load_state_dict`` intentionally casts floating state tensors
    to their parameter dtype. Prodigy+ScheduleFree stores some state (notably
    ``p0`` and ``s`` under stochastic rounding) in bf16 even when adapter
    parameters are fp32, so that generic behavior changes the resumed
    optimizer. Load group metadata normally, then replace each per-parameter
    state with a device-moved copy that retains its checkpoint dtype.
    """
    optimizer.load_state_dict(state_dict)

    saved_groups = state_dict.get("param_groups")
    saved_state = state_dict.get("state")
    if not isinstance(saved_groups, list) or not isinstance(saved_state, dict):
        raise TrainingStateError("cached optimizer state has an invalid structure")
    if len(saved_groups) != len(optimizer.param_groups):
        raise TrainingStateError("cached optimizer parameter-group count changed")

    parameter_map: dict[Any, torch.nn.Parameter] = {}
    for saved_group, current_group in zip(
        saved_groups, optimizer.param_groups, strict=True
    ):
        saved_params = saved_group.get("params")
        current_params = current_group.get("params")
        if not isinstance(saved_params, list) or len(saved_params) != len(current_params):
            raise TrainingStateError("cached optimizer parameter layout changed")
        parameter_map.update(zip(saved_params, current_params, strict=True))

    optimizer.state.clear()
    for saved_id, parameter_state in saved_state.items():
        parameter = parameter_map.get(saved_id)
        if parameter is None:
            raise TrainingStateError(
                f"cached optimizer state references unknown parameter {saved_id!r}"
            )
        optimizer.state[parameter] = _state_to_device(
            parameter_state, parameter.device
        )


def compatibility_mismatches(
    cached: dict[str, Any], expected: dict[str, Any]
) -> list[str]:
    mismatches: list[str] = []

    def compare(left: Any, right: Any, path: str) -> None:
        if isinstance(left, dict) and isinstance(right, dict):
            for key in sorted(set(left) | set(right)):
                child = f"{path}.{key}" if path else key
                if key not in left or key not in right:
                    mismatches.append(f"{child} changed")
                else:
                    compare(left[key], right[key], child)
        elif left != right:
            mismatches.append(f"{path} changed")

    compare(cached, expected, "")
    return mismatches


def progress_mismatches(
    manifest: dict[str, Any], *, max_train_epochs: int, steps_per_epoch: int
) -> list[str]:
    """Validate that an epoch-boundary checkpoint fits the requested run."""
    next_epoch = manifest["next_epoch"]
    global_step = manifest["global_step"]
    mismatches = []
    if next_epoch > max_train_epochs:
        mismatches.append(
            f"target epoch {max_train_epochs} precedes cached epoch {next_epoch}"
        )
    expected_step = next_epoch * steps_per_epoch
    if global_step != expected_step:
        mismatches.append(
            f"cached global step {global_step} does not match epoch boundary "
            f"{next_epoch} x {steps_per_epoch} = {expected_step}"
        )
    return mismatches


def _safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return cleaned[:64] or "run"


class TrainingStateCache:
    """Versioned training-state payloads associated with a LanceDB cache."""

    def __init__(
        self,
        cache_db: str | Path,
        *,
        output_dir: str | Path,
        output_name: str,
    ) -> None:
        identity = f"{Path(output_dir).resolve()}\0{output_name}"
        suffix = sha256(identity.encode("utf-8")).hexdigest()[:16]
        self.run_key = f"{_safe_name(output_name)}-{suffix}"
        self.run_dir = Path(cache_db) / "_training_state" / self.run_key
        self.manifest_path = self.run_dir / "latest.json"

    def read_manifest(self) -> dict[str, Any] | None:
        if not self.manifest_path.is_file():
            return None
        try:
            manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise TrainingStateError(f"cannot read {self.manifest_path}: {exc}") from exc
        if not isinstance(manifest, dict):
            raise TrainingStateError("cached state manifest is not a JSON object")
        required = {
            "format_version",
            "run_key",
            "generation",
            "payload_file",
            "payload_size",
            "payload_sha256",
            "compatibility",
            "compatibility_digest",
            "next_epoch",
            "global_step",
            "max_train_epochs",
            "saved_at_ns",
        }
        missing = required - manifest.keys()
        if missing:
            raise TrainingStateError(
                f"cached state manifest is missing fields: {sorted(missing)}"
            )
        if manifest["format_version"] != STATE_FORMAT_VERSION:
            raise TrainingStateError(
                "cached state format changed: "
                f"{manifest['format_version']} != {STATE_FORMAT_VERSION}"
            )
        if manifest["run_key"] != self.run_key:
            raise TrainingStateError("cached state belongs to a different output run")
        generation = manifest["generation"]
        if not isinstance(generation, str) or re.fullmatch(r"[0-9a-f]{32}", generation) is None:
            raise TrainingStateError("cached state generation is invalid")
        payload_name = manifest["payload_file"]
        if (
            not isinstance(payload_name, str)
            or Path(payload_name).name != payload_name
            or payload_name != f"state-{generation}.pt"
        ):
            raise TrainingStateError("cached state manifest contains an unsafe payload path")
        if (
            not isinstance(manifest["payload_size"], int)
            or isinstance(manifest["payload_size"], bool)
            or manifest["payload_size"] <= 0
        ):
            raise TrainingStateError("cached state payload size is invalid")
        payload_sha256 = manifest["payload_sha256"]
        if not isinstance(payload_sha256, str) or re.fullmatch(
            r"[0-9a-f]{64}", payload_sha256
        ) is None:
            raise TrainingStateError("cached state payload digest is invalid")
        if not isinstance(manifest["compatibility"], dict):
            raise TrainingStateError("cached compatibility contract is not an object")
        if (
            not isinstance(manifest["next_epoch"], int)
            or isinstance(manifest["next_epoch"], bool)
            or manifest["next_epoch"] < 0
        ):
            raise TrainingStateError("cached next epoch is invalid")
        if (
            not isinstance(manifest["global_step"], int)
            or isinstance(manifest["global_step"], bool)
            or manifest["global_step"] < 0
        ):
            raise TrainingStateError("cached global step is invalid")
        if (
            not isinstance(manifest["max_train_epochs"], int)
            or isinstance(manifest["max_train_epochs"], bool)
            or manifest["max_train_epochs"] <= 0
        ):
            raise TrainingStateError("cached target epoch is invalid")
        if (
            not isinstance(manifest["saved_at_ns"], int)
            or isinstance(manifest["saved_at_ns"], bool)
            or manifest["saved_at_ns"] <= 0
        ):
            raise TrainingStateError("cached save timestamp is invalid")
        digest = compatibility_digest(manifest["compatibility"])
        if (
            not isinstance(manifest["compatibility_digest"], str)
            or digest != manifest["compatibility_digest"]
        ):
            raise TrainingStateError("cached compatibility manifest failed its digest check")
        return manifest

    def load_payload(self, manifest: dict[str, Any]) -> dict[str, Any]:
        payload_name = manifest["payload_file"]
        if (
            not isinstance(payload_name, str)
            or Path(payload_name).name != payload_name
            or not payload_name.startswith("state-")
            or not payload_name.endswith(".pt")
        ):
            raise TrainingStateError("cached state manifest contains an unsafe payload path")
        path = self.run_dir / payload_name
        try:
            if path.stat().st_size != manifest["payload_size"]:
                raise TrainingStateError(f"cached state payload size check failed: {path}")
            if _file_sha256(path) != manifest["payload_sha256"]:
                raise TrainingStateError(f"cached state payload digest check failed: {path}")
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except TrainingStateError:
            raise
        except Exception as exc:
            raise TrainingStateError(f"cannot load cached training state {path}: {exc}") from exc
        if not isinstance(payload, dict):
            raise TrainingStateError("cached training payload is not a mapping")
        missing = {"network", "optimizer", "rng"} - payload.keys()
        if missing:
            raise TrainingStateError(
                f"cached training payload is missing fields: {sorted(missing)}"
            )
        for field in ("network", "optimizer", "rng"):
            if not isinstance(payload[field], dict):
                raise TrainingStateError(
                    f"cached training payload field {field!r} is not a mapping"
                )
        if payload.get("format_version") != STATE_FORMAT_VERSION:
            raise TrainingStateError("cached payload format does not match the manifest")
        if payload.get("run_key") != self.run_key:
            raise TrainingStateError("cached payload belongs to a different output run")
        if payload.get("generation") != manifest["generation"]:
            raise TrainingStateError("cached payload generation does not match the manifest")
        if payload.get("compatibility_digest") != manifest["compatibility_digest"]:
            raise TrainingStateError("cached payload compatibility does not match the manifest")
        if payload.get("next_epoch") != manifest["next_epoch"]:
            raise TrainingStateError("cached payload epoch does not match the manifest")
        if payload.get("global_step") != manifest["global_step"]:
            raise TrainingStateError("cached payload step does not match the manifest")
        return payload

    def save(
        self,
        *,
        network: torch.nn.Module,
        optimizer,
        next_epoch: int,
        global_step: int,
        max_train_epochs: int,
        compatibility: dict[str, Any],
    ) -> Path:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        generation = uuid4().hex
        payload_name = f"state-{generation}.pt"
        payload_path = self.run_dir / payload_name
        temp_payload = self.run_dir / f".{payload_name}.tmp-{os.getpid()}"
        digest = compatibility_digest(compatibility)
        saved_at_ns = time.time_ns()
        payload = {
            "format_version": STATE_FORMAT_VERSION,
            "run_key": self.run_key,
            "generation": generation,
            "compatibility_digest": digest,
            "next_epoch": int(next_epoch),
            "global_step": int(global_step),
            "network": _to_cpu_snapshot(network.state_dict()),
            "optimizer": _to_cpu_snapshot(optimizer.state_dict()),
            "rng": capture_rng_state(),
        }

        previous = self.read_manifest()
        try:
            with temp_payload.open("wb") as handle:
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_payload, payload_path)

            manifest = {
                "format_version": STATE_FORMAT_VERSION,
                "run_key": self.run_key,
                "generation": generation,
                "payload_file": payload_name,
                "payload_size": payload_path.stat().st_size,
                "payload_sha256": _file_sha256(payload_path),
                "compatibility": compatibility,
                "compatibility_digest": digest,
                "next_epoch": int(next_epoch),
                "global_step": int(global_step),
                "max_train_epochs": int(max_train_epochs),
                "saved_at_ns": saved_at_ns,
            }
            self._write_manifest(manifest)
        finally:
            temp_payload.unlink(missing_ok=True)

        if previous is not None and previous.get("payload_file") != payload_name:
            old_name = previous.get("payload_file")
            if isinstance(old_name, str) and Path(old_name).name == old_name:
                (self.run_dir / old_name).unlink(missing_ok=True)
        self._fsync_dir()
        return payload_path

    def _write_manifest(self, manifest: dict[str, Any]) -> None:
        temp = self.run_dir / f".latest.tmp-{os.getpid()}-{uuid4().hex}"
        try:
            with temp.open("w", encoding="utf-8") as handle:
                json.dump(manifest, handle, sort_keys=True, separators=(",", ":"))
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp, self.manifest_path)
            self._fsync_dir()
        finally:
            temp.unlink(missing_ok=True)

    def quarantine(self, reason: str) -> Path | None:
        if not self.manifest_path.exists():
            return None
        suffix = f"{time.time_ns()}-{_safe_name(reason)[:32]}"
        destination = self.run_dir / f"invalid-{suffix}.json"
        os.replace(self.manifest_path, destination)
        self._fsync_dir()
        return destination

    def clear(self) -> None:
        manifest = self.read_manifest()
        if manifest is None:
            return
        payload_name = manifest["payload_file"]
        self.manifest_path.unlink(missing_ok=True)
        if Path(payload_name).name == payload_name:
            (self.run_dir / payload_name).unlink(missing_ok=True)
        self._fsync_dir()

    def _fsync_dir(self) -> None:
        try:
            fd = os.open(self.run_dir, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        except OSError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def choose_resume(
    manifest: dict[str, Any],
    *,
    requested: bool | None,
    input_fn: Callable[[str], str] = input,
    is_interactive: bool | None = None,
) -> bool:
    """Resolve CLI override or ask before restoring a compatible state."""
    if requested is not None:
        return requested
    if is_interactive is None:
        is_interactive = sys.stdin.isatty()
    if not is_interactive:
        raise TrainingStateError(
            "compatible cached training state is available at epoch "
            f"{manifest['next_epoch']}; non-interactive runs must pass "
            "--resume or --no-resume"
        )
    completed = int(manifest["next_epoch"])
    step = int(manifest["global_step"])
    answer = input_fn(
        f"Resume cached training state after epoch {completed} "
        f"(global step {step})? [Y/n] "
    ).strip().lower()
    if answer in ("", "y", "yes"):
        return True
    if answer in ("n", "no"):
        return False
    raise TrainingStateError("expected 'y' or 'n' at the resume prompt")
