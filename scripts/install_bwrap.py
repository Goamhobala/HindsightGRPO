"""Select or build a pinned scratch-local bubblewrap without sudo or Git.

`--check` only inspects command options/version and is safe on the login node.
Namespace and worker execution are still tested separately on a compute node.
"""
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import urllib.request

from hindsight_grpo.rewards import SandboxUnavailable, check_bubblewrap_capabilities


BWRAP_VERSION = "0.13.0"
BWRAP_SHA256 = "4734237473c0e5d695e4e9034a34e43b2dbf5164655bd13fa59ae376b2b7a765"
BWRAP_URL = f"https://github.com/containers/bubblewrap/releases/download/v{BWRAP_VERSION}/bubblewrap-{BWRAP_VERSION}.tar.xz"
LIBCAP_VERSION = "2.75"
LIBCAP_SHA256 = "de4e7e064c9ba451d5234dd46e897d7c71c96a9ebf9a0c445bc04f4742d83632"
LIBCAP_URL = f"https://www.kernel.org/pub/linux/libs/security/linux-privs/libcap2/libcap-{LIBCAP_VERSION}.tar.xz"
MESON_VERSION = "1.9.1"
NINJA_VERSION = "1.13.0"


def verified_download(url: str, destination: Path, expected_sha256: str) -> Path:
    """Cache only a digest-verified official source archive; never execute downloads."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and hashlib.sha256(destination.read_bytes()).hexdigest() == expected_sha256:
        return destination
    descriptor, name = tempfile.mkstemp(prefix=".download-", dir=destination.parent)
    temporary = Path(name)
    try:
        digest = hashlib.sha256()
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(url, timeout=60) as response:
            total = 0
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > 32 * 1024 * 1024:
                    raise RuntimeError("Source archive exceeded the 32 MiB download limit")
                digest.update(block)
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        if digest.hexdigest() != expected_sha256:
            raise RuntimeError(f"SHA-256 mismatch for {url}; source was not extracted or built")
        os.replace(temporary, destination)
        return destination
    finally:
        temporary.unlink(missing_ok=True)


def extract_source(archive_path: Path, destination: Path, expected_root: str) -> Path:
    """Restrict all extracted paths to the single expected source directory."""
    with tarfile.open(archive_path, "r:xz") as archive:
        for member in archive.getmembers():
            path = PurePosixPath(member.name)
            if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != expected_root:
                raise RuntimeError(f"Unexpected source archive path: {member.name}")
            if not (member.isfile() or member.isdir() or member.issym() or member.islnk()):
                raise RuntimeError(f"Unexpected source archive member: {member.name}")
        # Python's data filter rejects escaping links, devices and unsafe modes.
        archive.extractall(destination, filter="data")
    return destination / expected_root


def build_prerequisites() -> dict[str, str]:
    programs = {name: shutil.which(name) for name in ("uv", "cc", "make", "ar", "ranlib", "pkg-config")}
    missing = [name for name, path in programs.items() if not path]
    if missing:
        raise RuntimeError("Building scratch-local bubblewrap requires these build tools: "
                           + ", ".join(missing) + ". Load the cluster compiler/build-tools module and rerun. "
                           "No system libcap development package or sudo is needed.")
    return programs


def install(runtime: Path) -> dict:
    """Build verified libcap statically, then the unprivileged bubblewrap binary."""
    if platform.system() != "Linux":
        raise RuntimeError("Build bubblewrap on the Linux HPC login node")
    programs = build_prerequisites()
    runtime = runtime.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    cache = runtime / "downloads"
    bwrap_archive = verified_download(BWRAP_URL, cache / f"bubblewrap-{BWRAP_VERSION}.tar.xz", BWRAP_SHA256)
    cap_archive = verified_download(LIBCAP_URL, cache / f"libcap-{LIBCAP_VERSION}.tar.xz", LIBCAP_SHA256)
    tools_dir = runtime / "bwrap-build-venv"
    uv_environment = dict(os.environ, UV_CACHE_DIR=str(runtime / "uv-cache"))
    subprocess.run([programs["uv"], "venv", "--allow-existing", "--python", sys.executable, str(tools_dir)],
                   check=True, env=uv_environment)
    subprocess.run([programs["uv"], "pip", "install", "--python", str(tools_dir / "bin/python"),
                    f"meson=={MESON_VERSION}", f"ninja=={NINJA_VERSION}"], check=True, env=uv_environment)
    jobs = min(8, max(1, int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))))
    prefix = runtime / "opt" / f"bubblewrap-{BWRAP_VERSION}"
    cap_prefix = runtime / "opt" / f"libcap-{LIBCAP_VERSION}"
    build_parent = runtime / "build"
    build_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="bubblewrap-", dir=build_parent) as name:
        working = Path(name)
        cap_source = extract_source(cap_archive, working, f"libcap-{LIBCAP_VERSION}")
        cap_arguments = [f"CC={programs['cc']}", f"BUILD_CC={programs['cc']}",
                         f"AR={programs['ar']}", f"RANLIB={programs['ranlib']}",
                         "SHARED=no", "PTHREADS=no", "GOLANG=no", "PAM_CAP=no", "USE_GPERF=no",
                         f"prefix={cap_prefix}", "lib=lib"]
        subprocess.run([programs["make"], "-C", str(cap_source / "libcap"), f"-j{jobs}",
                        *cap_arguments, "all"], check=True)
        subprocess.run([programs["make"], "-C", str(cap_source / "libcap"),
                        *cap_arguments, "install-static-cap"], check=True)
        source = extract_source(bwrap_archive, working, f"bubblewrap-{BWRAP_VERSION}")
        build = working / "bwrap-build"
        environment = dict(os.environ)
        environment["PATH"] = str(tools_dir / "bin") + os.pathsep + environment.get("PATH", "")
        environment["CC"] = programs["cc"]
        environment["PKG_CONFIG"] = programs["pkg-config"]
        environment["PKG_CONFIG_PATH"] = str(cap_prefix / "lib/pkgconfig") + os.pathsep + environment.get("PKG_CONFIG_PATH", "")
        meson = str(tools_dir / "bin/meson")
        subprocess.run([meson, "setup", str(build), str(source), f"--prefix={prefix}",
                        "--buildtype=release", "-Dprefer_static=true", "-Dselinux=disabled",
                        "-Dman=disabled", "-Dtests=false", "-Dbash_completion=disabled",
                        "-Dzsh_completion=disabled"], check=True, env=environment)
        subprocess.run([meson, "compile", "-C", str(build), "-j", str(jobs)], check=True, env=environment)
        subprocess.run([meson, "install", "-C", str(build), "--no-rebuild"], check=True, env=environment)
    executable = prefix / "bin/bwrap"
    capabilities = check_bubblewrap_capabilities(str(executable))
    binaries = runtime / "bin"
    binaries.mkdir(parents=True, exist_ok=True)
    link = binaries / ".bwrap-new"
    link.unlink(missing_ok=True)
    link.symlink_to(os.path.relpath(executable, binaries))
    os.replace(link, binaries / "bwrap")
    metadata = {**capabilities, "installed_binary": str(binaries / "bwrap"),
                "bubblewrap_source": {"url": BWRAP_URL, "sha256": BWRAP_SHA256},
                "libcap_source": {"url": LIBCAP_URL, "sha256": LIBCAP_SHA256},
                "libcap_linkage": "static", "meson": MESON_VERSION, "ninja": NINJA_VERSION,
                "setuid": False}
    metadata_dir = runtime / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    temporary_manifest = metadata_dir / "bubblewrap.json.tmp"
    temporary_manifest.write_text(json.dumps(metadata, indent=2) + "\n")
    os.replace(temporary_manifest, metadata_dir / "bubblewrap.json")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-dir", default="runtime")
    parser.add_argument("--check", action="store_true", help="Inspect the selected binary only; do not build or run a worker")
    parser.add_argument("--force", action="store_true", help="Build the pinned binary even if the current binary supports the policy")
    args = parser.parse_args()
    if args.check:
        print(json.dumps(check_bubblewrap_capabilities(), indent=2))
        return
    if not args.force:
        try:
            print(json.dumps(check_bubblewrap_capabilities(), indent=2))
            return
        except SandboxUnavailable as error:
            print(str(error), file=sys.stderr)
    runtime = Path(args.runtime_dir).resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    with (runtime / ".bubblewrap-install.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        result = install(runtime)
    override = os.environ.get("HINDSIGHT_BWRAP")
    if override and Path(override).expanduser().resolve() != Path(result["installed_binary"]).resolve():
        raise RuntimeError("The pinned build is ready, but HINDSIGHT_BWRAP still selects another binary. "
                           "Unset HINDSIGHT_BWRAP or point it to " + result["installed_binary"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (SandboxUnavailable, RuntimeError, subprocess.CalledProcessError, OSError) as error:
        raise SystemExit(str(error))
