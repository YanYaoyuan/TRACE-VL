"""Install the locked Linux/Python 3.10 environment through Aliyun mirrors."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ALIYUN = "https://mirrors.aliyun.com/pypi/simple"
PYPI = "https://pypi.org/simple"
CANONICAL_CUDA = "https://download.pytorch.org/whl/cu128"
CUDA = "https://mirrors.aliyun.com/pytorch-wheels/cu128"
URL_REPLACEMENTS = (
    ("https://files.pythonhosted.org/packages/", "https://mirrors.aliyun.com/pypi/packages/"),
    ("https://download-r2.pytorch.org/whl/cu128/", CUDA + "/"),
    (CANONICAL_CUDA, CUDA),
    (PYPI, ALIYUN),
)
CUDA_WHEELS = {
    "torch": CUDA + "/torch-2.8.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl",
    "torchvision": CUDA + "/torchvision-0.23.0%2Bcu128-cp310-cp310-manylinux_2_28_x86_64.whl",
}


def remap(text: str) -> str:
    for old, new in URL_REPLACEMENTS:
        text = text.replace(old, new)
    return text


def inventory(lock: str) -> dict[tuple[str, str], str | None]:
    packages = {}
    for block in lock.split("[[package]]")[1:]:
        name = re.search(r'^name = "([^"]+)"$', block, re.M)
        version = re.search(r'^version = "([^"]+)"$', block, re.M)
        source = re.search(r'^source = \{ registry = "([^"]+)" \}$', block, re.M)
        if name and version:
            packages[name[1], version[1]] = source[1] if source else None
    if not packages:
        raise ValueError("No packages found in uv.lock")
    return packages


def verify_lock(canonical: str, mirrored: str) -> dict:
    # Exact textual remapping also preserves all dependency edges, platform markers and artifact hashes.
    if mirrored != remap(canonical):
        raise ValueError("Mirror lock must change download/index URLs only; versions, markers and hashes must match")
    old, new = inventory(canonical), inventory(mirrored)
    if set(old) != set(new):
        raise ValueError("Mirror changed the locked package versions")
    for key, registry in new.items():
        if registry is not None and registry not in {ALIYUN, CUDA}:
            raise ValueError(f"Unexpected registry for {key[0]}: {registry}")
    for name, version in (("torch", "2.8.0+cu128"), ("torchvision", "0.23.0+cu128")):
        if new.get((name, version)) != CUDA or CUDA_WHEELS[name] not in mirrored:
            raise ValueError(f"{name} must retain the pinned CUDA 12.8 build on Aliyun")
    for url in re.findall(r'https://[^"\s]+', mirrored):
        if not url.startswith("https://mirrors.aliyun.com/"):
            raise ValueError(f"Unmapped download URL: {url}")
    return {name: sorted({version for n, version in new if n == name}) for name, _ in new}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Audit the mirror lock and install plan; do not change .venv")
    parser.add_argument("--output", type=Path, help="Audit directory (default: outputs/environments/aliyun)")
    args = parser.parse_args()
    if sys.platform != "linux" or platform.machine().lower() != "x86_64":
        parser.error("This validated installer targets Linux x86_64 with Python 3.10")
    root = Path(__file__).resolve().parents[1]
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv before running this script")
    if (root / ".git").exists():
        dirty = subprocess.run(["git", "-C", str(root), "diff", "--quiet", "HEAD", "--", "pyproject.toml", "uv.lock"])
        if dirty.returncode:
            parser.error("Dependency files have local changes. Back them up and restore the committed files before retrying.")
    canonical = {name: (root / name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
    lock = canonical["uv.lock"].decode()
    mirrored = remap(lock)
    versions = verify_lock(lock, mirrored)
    output = (args.output or root / "outputs/environments/aliyun").resolve()
    output.mkdir(parents=True, exist_ok=True)
    (output / "provenance.json").unlink(missing_ok=True)
    env = os.environ.copy()
    for name in ("UV_INDEX", "UV_INDEX_URL", "UV_DEFAULT_INDEX", "UV_EXTRA_INDEX_URL", "UV_FIND_LINKS",
                 "UV_NO_SOURCES", "UV_INDEX_STRATEGY", "UV_CONFIG_FILE", "UV_PROJECT", "UV_WORKING_DIR",
                 "UV_FROZEN", "UV_LOCKED", "UV_CONSTRAINT", "UV_BUILD_CONSTRAINT", "UV_NO_CONFIG"):
        env.pop(name, None)
    env["UV_PROJECT_ENVIRONMENT"] = str(root / ".venv")
    env.setdefault("UV_HTTP_TIMEOUT", "300")
    installed = None
    try:
        with tempfile.TemporaryDirectory(prefix="install-", dir=output) as scratch:
            project = Path(scratch)
            (project / "pyproject.toml").write_text(remap(canonical["pyproject.toml"].decode()))
            (project / "uv.lock").write_text(mirrored)
            (project / "README.md").write_text("Temporary mirror dependency project.\n")
            for name in ("uv.lock", "pyproject.toml"):
                shutil.copyfile(project / name, output / name)
            # The CUDA mirror is a flat wheel directory; frozen sync uses the exact wheel URLs, not a
            # PEP 503 /torch/ endpoint. No re-resolution or package upgrades are needed.
            command = [uv, "sync", "--project", str(project), "--frozen", "--python", "3.10",
                       "--extra", "dev", "--extra", "vlm", "--extra", "data", "--extra", "track",
                       "--no-install-project"]
            if args.dry_run:
                command.append("--dry-run")
            subprocess.run(command, env=env, check=True)
        if not args.dry_run:
            subprocess.run([uv, "pip", "install", "--python", str(root / ".venv/bin/python"), "--no-deps",
                            "--index-url", ALIYUN, "--editable", str(root)], env=env, check=True)
            subprocess.run([uv, "pip", "check", "--python", str(root / ".venv/bin/python")], env=env, check=True)
            probe = "import importlib.metadata as m,json; print(json.dumps({d.metadata['Name'].lower().replace('_','-'):d.version for d in m.distributions()}))"
            result = subprocess.run([str(root / ".venv/bin/python"), "-I", "-c", probe], env=env, check=True, text=True, capture_output=True)
            installed = json.loads(result.stdout)
            for name, version in installed.items():
                if (name, version) not in inventory(lock):
                    raise ValueError(f"Installed version is absent from canonical lock: {name}=={version}")
        report = {"mirror": ALIYUN, "cuda_mirror": CUDA, "python": "3.10", "dry_run": args.dry_run,
                  "locked_versions": versions, "installed_versions": installed,
                  "canonical_sha256": {name: hashlib.sha256(data).hexdigest() for name, data in canonical.items()}}
        (output / "provenance.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"Mirror setup {'dry run' if args.dry_run else 'installation'} completed; audit: {output}")
    finally:
        for name, before in canonical.items():
            if (root / name).read_bytes() != before:
                raise RuntimeError(f"Canonical dependency file changed unexpectedly: {name}")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, subprocess.CalledProcessError) as error:
        print(f"Mirror setup failed: {error}", file=sys.stderr)
        raise SystemExit(1) from error
