import argparse
import re
from pathlib import Path

from mkdocstrings_parser import MkDocstringsParser

comment_pat = re.compile(r"<!--.*?-->", re.DOTALL)
anchor_pat = re.compile(r"<a.*?>(.*?)</a>")
output_path = Path("docs/mintlify")


def process_files(input_dir):
    """Process files with MkDocstrings parser, then clean with regex"""
    # Step 1: Use MkDocstrings parser to generate initial MDX files
    parser = MkDocstringsParser()
    for file in Path(input_dir).rglob("*.md"):
        folder_path = Path(input_dir) / "mintlify" / Path(*file.parent.parts[1:])
        folder_path.mkdir(parents=True, exist_ok=True)
        output_file = str(folder_path / file.with_suffix(".mdx").name)
        print(f"Processing {file} -> {output_file}")
        parser.process_file(str(file), output_file)

    # Step 2: Clean up the generated MDX files with regex patterns
    for mdx_file in (Path(input_dir) / "mintlify").glob("*.mdx"):
        if mdx_file.name == "index.mdx":  # Skip index.mdx as it's handled separately
            continue
        print(f"Cleaning up {mdx_file}")
        text = mdx_file.read_text()
        text = comment_pat.sub("", text)
        text = anchor_pat.sub("", text)
        mdx_file.write_text(text)


def copy_readme():
    """Copy README.md to index.mdx with proper header"""
    header = """---
description: Lightning fast forecasting with statistical and econometric models
title: "Statistical ⚡️ Forecast"
---
"""
    readme_text = Path("README.md").read_text()
    # Skip the first 22 lines
    lines = readme_text.split('\n')
    readme_text = '\n'.join(lines[22:])
    readme_text = header + readme_text
    (output_path / "index.html.mdx").write_text(readme_text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process markdown files to MDX format")
    parser.add_argument(
        "input_dir", nargs="?", default="docs", help="Input directory (default: docs)"
    )
    args = parser.parse_args()

    # Step 1: Process files with MkDocstrings parser, then clean with regex
    process_files(args.input_dir)

    # Step 2: Always copy the README
    copy_readme()
