"""Standard-library-only benchmark file handling, including Modal local launch."""

import hashlib
import json
from pathlib import Path


def source_identity(root):
    files = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
             for p in sorted((root / "src").rglob("*.py"))}
    for name in ("pyproject.toml", "uv.lock"):
        files[name] = hashlib.sha256((root / name).read_bytes()).hexdigest()
    return {"files": files, "sha256": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()}


def frozen_inputs(root, directory=None):
    directory = Path(directory) if directory is not None else Path("workloads/baseline")
    if not directory.is_absolute():
        directory = root / directory
    texts = {name: (directory / name).read_text() for name in
             ("mixed-v1.json", "manifest.json", "prompt-bank.json")}
    manifest = json.loads(texts["manifest.json"])
    for key in ("workload", "prompt_bank"):
        entry = manifest[key]
        assert hashlib.sha256(texts[entry["file"]].encode()).hexdigest() == entry["sha256"], "Frozen input checksum mismatch"
    return texts


def save_report(report, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    for run in report["runs"]:
        (directory / f"run-{run['run']:02d}.json").write_text(json.dumps(run, indent=2, allow_nan=False) + "\n")
    (directory / "baseline.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
