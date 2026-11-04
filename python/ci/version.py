#!/usr/bin/env python3
# the version of the wheel: major.minor from pyproject.toml (version = "3.0"), the third number is the number of commits
# since the commit that changed that line (3.0.0 there, then +1 every commit). Prints it; with --write also writes it
# in pyproject.toml (CI, before cibuildwheel: not to be committed)
import pathlib
import re
import subprocess
import sys

root = pathlib.Path(__file__).resolve().parents[2]
pyproject = root / "pyproject.toml"
# newline="": keeps the line endings (CRLF in the Windows checkout)
with open(pyproject, newline="") as f:
    text = f.read()

line = re.compile(r'^version = "(\d+)\.(\d+)(?:\.\d+)?"', re.M)
m = line.search(text)
if not m:
    sys.exit(f'no version = "major.minor" in {pyproject}')


def git(*args):
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=True).stdout.strip()


# the last commit that changed the version line (needs the whole history: checkout with fetch-depth: 0)
base = git("log", "-n1", "--format=%H", "-G", '^version = "', "--", "pyproject.toml")
patch = git("rev-list", "--count", f"{base}..HEAD") if base else "0"
version = f"{m.group(1)}.{m.group(2)}.{patch}"

if "--write" in sys.argv[1:]:
    with open(pyproject, "w", newline="") as f:
        f.write(line.sub(f'version = "{version}"', text, count=1))
print(version)
