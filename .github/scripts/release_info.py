"""Check a release tag against the package versions and print release metadata.

Usage: python .github/scripts/release_info.py v0.2.0
Fails unless every package's version equals the tag (without the leading "v").
Writes ``lerobot_spec`` (the robot plugin's lerobot requirement) and
``lerobot_tested`` (its lower bound, the release CI gates on) to $GITHUB_OUTPUT.
"""

from __future__ import annotations

import os
import re
import sys
import tomllib
from pathlib import Path

PACKAGES = ["lerobot-robot-rby1", "lerobot-teleoperator-rby1", "lerobot-async-rby1"]


def main(tag: str) -> int:
    root = Path(__file__).resolve().parents[2]
    expected = tag.removeprefix("v")
    errors = []
    specs: dict[str, str] = {}
    for pkg in PACKAGES:
        project = tomllib.loads((root / pkg / "pyproject.toml").read_text())["project"]
        if project["version"] != expected:
            errors.append(f"{pkg}: version {project['version']} != tag {tag}")
        for dep in project["dependencies"]:
            if re.match(r"lerobot\s*[<>=!~]", dep):
                specs[pkg] = dep
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1

    spec = specs["lerobot-robot-rby1"]
    lower = re.search(r">=\s*([0-9][0-9a-z.]*)", spec)
    if not lower:
        print(f"no lower bound in {spec!r}", file=sys.stderr)
        return 1
    out = {"lerobot_spec": spec, "lerobot_tested": lower.group(1), "async_spec": specs["lerobot-async-rby1"]}
    for key, value in out.items():
        print(f"{key}={value}")
    if "GITHUB_OUTPUT" in os.environ:
        with open(os.environ["GITHUB_OUTPUT"], "a") as fh:
            fh.writelines(f"{key}={value}\n" for key, value in out.items())
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1]))
