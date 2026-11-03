"""Installs the official Qt for Windows from download.qt.io (for the python wheel: the Qt of PySide6).

    python install_qt_windows.py 6.12.0 msvc2022_64 C:\\Qt qtcharts qtmultimedia

From Qt 6.12 the Windows repository has a folder per compiler (qt6_6120/qt6_6120_msvc2022_64/Updates.xml), not
known by aqtinstall 3.3 (it looks for qt6_6120/qt6_6120/Updates.xml). The archives are checked (sha1) and extracted
in <outdir>/<version>/<arch> (e.g. C:\\Qt\\6.12.0\\msvc2022_64), the folder to give to CMAKE_PREFIX_PATH.
Needs py7zr (pip install py7zr, it comes with aqtinstall)."""
import hashlib
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import py7zr

REPO = "https://download.qt.io/online/qtsdkrepository/windows_x86/desktop"
# archives of the base package not needed to compile (documentation, software OpenGL, ...)
SKIP = ("qtdoc", "opengl32sw", "d3dcompiler")


def download(url, retries=3):
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                return response.read()
        except OSError as e:
            if attempt == retries - 1:
                raise
            print(f"retry {url}: {e}", flush=True)


def main(version, arch, outdir, *modules):
    v = version.replace(".", "")
    base = f"{REPO}/qt6_{v}/qt6_{v}_{arch}"
    updates = ET.fromstring(download(f"{base}/Updates.xml"))
    wanted = [f"qt.qt6.{v}.win64_{arch}"] + [f"qt.qt6.{v}.addons.{m}.win64_{arch}" for m in modules]
    packages = {pu.findtext("Name"): pu for pu in updates.findall("PackageUpdate")}
    # the archives have no <version>/<arch> prefix (aqt archives had it)
    target = Path(outdir) / version / arch
    target.mkdir(parents=True, exist_ok=True)
    for name in wanted:
        if name not in packages:
            sys.exit(f"{name} not in {base}/Updates.xml")
        pu = packages[name]
        archives = [a.strip() for a in (pu.findtext("DownloadableArchives") or "").split(",") if a.strip()]
        for archive in archives:
            if archive.startswith(SKIP):
                continue
            url = f"{base}/{name}/{pu.findtext('Version')}{archive}"
            print(f"{name}: {archive}", flush=True)
            data = download(url)
            sha1 = download(url + ".sha1").decode().split()[0]
            if hashlib.sha1(data).hexdigest() != sha1:
                sys.exit(f"sha1 mismatch: {url}")
            tmp = target / archive
            tmp.write_bytes(data)
            with py7zr.SevenZipFile(tmp) as z:
                z.extractall(target)
            tmp.unlink()
    print(f"Qt in {target}")


if __name__ == "__main__":
    main(*sys.argv[1:])
