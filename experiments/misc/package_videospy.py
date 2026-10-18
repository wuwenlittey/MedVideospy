#!/usr/bin/env python3
"""Package tracked VideoSpy sources and sanitized configs for supplementary material.

Run from any directory: python experiments/misc/package_videospy.py [--output FILE.zip]
Requires PyYAML. Reads current working-tree contents, without changing local configs.
"""

import argparse
from pathlib import Path
import subprocess
from zipfile import ZIP_DEFLATED, ZipFile

import yaml


ROOT = Path(__file__).resolve().parents[2]
README = ROOT / "docs/research/evolving_video_agent/iclr2027/code_README.md"
PACKAGE_NAME = "videospy-code"
PYPROJECT = '''[build-system]
requires = ["setuptools>=61.0"]
build-backend = "setuptools.build_meta"

[project]
name = "videospy"
version = "0.1.0"
description = "VideoSpy core agent and experience evolution"
readme = "README.md"
requires-python = ">=3.10"
dependencies = ["decord", "litellm", "openai", "numpy", "pillow", "tqdm", "pyyaml"]

[project.scripts]
videospy-cli = "videospy.cli:main"

[tool.setuptools.packages.find]
include = ["videospy*", "config*", "benchmark*"]

[tool.setuptools.package-data]
"config" = ["*.yml"]
"config.videospy" = ["*.yml", "*.yaml"]
'''


class ConfigDumper(yaml.SafeDumper):
    """Keep multiline prompts readable in the exported YAML."""


ConfigDumper.add_representer(
    str,
    lambda dumper, value: dumper.represent_scalar(
        "tag:yaml.org,2002:str", value, style="|" if "\n" in value else None
    ),
)


def sanitize_config(value, path=()):
    if isinstance(value, list):
        return [sanitize_config(item, path) for item in value]
    if not isinstance(value, dict):
        return value
    result = {}
    prefix = "VIDEOSPY_EMBEDDING" if "embedding_model" in path else "VIDEOSPY"
    for key, item in value.items():
        if key in {"api_key", "api_base"}:
            result[key] = f"{prefix}_{key.upper()}"
        elif key == "model_name":
            result[key] = f"{prefix}_MODEL"
        elif key in {"api_version", "database_path", "version_id"}:
            result[key] = None
        else:
            result[key] = sanitize_config(item, path + (key,))
    return result


def package(output: Path) -> Path:
    paths = subprocess.check_output(
        ["git", "ls-files", "-z", "--", "videospy", "config/videospy",
         "benchmark/sampling.py", "config/benchmark_sampling.yml"],
        cwd=ROOT,
    ).decode().split("\0")
    contents = {}
    for name in sorted(filter(None, paths)):
        source = ROOT / name
        if source.suffix not in {".py", ".yml", ".yaml"}:
            continue
        if source.is_symlink():
            raise ValueError(f"Refusing to package a symlink: {name}")
        if source.suffix in {".yml", ".yaml"}:
            config = sanitize_config(yaml.safe_load(source.read_text(encoding="utf-8")))
            contents[name] = yaml.dump(
                config, Dumper=ConfigDumper, sort_keys=False, allow_unicode=True
            ).encode("utf-8")
        else:
            contents[name] = source.read_bytes()

    # The repository's config initializer imports VideoSeek, which is not bundled.
    contents["config/__init__.py"] = b'"""Configuration for the standalone VideoSpy package."""\n'
    # Evolution imports the shared sampling helper; no benchmark runners are bundled.
    contents["benchmark/__init__.py"] = b'"""Shared sampling support for VideoSpy evolution."""\n'
    contents["pyproject.toml"] = PYPROJECT.encode("utf-8")
    contents["LICENSE"] = (ROOT / "LICENSE").read_bytes()
    contents["README.md"] = README.read_bytes()
    contents["videospy/README.md"] = contents["README.md"]

    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, data in sorted(contents.items()):
            archive.writestr(f"{PACKAGE_NAME}/{name}", data)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path,
        default=ROOT / "output/supplementary/videospy-code.zip",
        help="Destination ZIP (default: output/supplementary/videospy-code.zip).",
    )
    args = parser.parse_args()
    output = package(args.output)
    print(f"Created {output} ({output.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
