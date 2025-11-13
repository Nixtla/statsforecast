import argparse
import logging
import re
from typing import Any, Dict, Optional

import griffe
import yaml
from griffe2md import ConfigDict, render_object_docs

# from rich.console import Console
# from rich.markdown import Markdown

# Suppress griffe warnings
logging.getLogger("griffe").setLevel(logging.ERROR)


class MkDocstringsParser:
    def __init__(self):
        pass

    def parse_docstring_block(
        self, block_content: str
    ) -> tuple[str, str, Dict[str, Any]]:
        """Parse a ::: block to extract module path, handler, and options"""
        lines = block_content.strip().split("\n")

        # First line contains the module path
        module_path = lines[0].replace(":::", "").strip()

        # Parse YAML configuration
        yaml_content = "\n".join(lines[1:]) if len(lines) > 1 else ""

        try:
            config = yaml.safe_load(yaml_content) or {}
        except yaml.YAMLError:
            config = {}

        handler_type = config.get("handler", "python")
        options = config.get("options", {})

        return module_path, handler_type, options

    def generate_documentation(self, module_path: str, options: Dict[str, Any]) -> str:
        """Generate documentation for a given module using griffe and griffe2md"""
        try:
            # Parse the module path to extract package and object path
            if "." in module_path:
                parts = module_path.split(".")
                package_name = parts[0]
                object_path = ".".join(parts[1:])
            else:
                package_name = module_path
                object_path = ""

            # Load the package with griffe
            package = griffe.load(package_name)
            to_replace = ".".join((package_name + "." + object_path).split(".")[:-1])
            # Get the specific object if path is provided
            if object_path:
                obj = package[object_path]
            else:
                obj = package

            # Ensure the docstring is properly parsed with Google parser
            # For functions, we might need to get the actual runtime docstring
            if hasattr(obj, "kind") and obj.kind.value == "function":
                try:
                    # Try to get the actual function object to access runtime docstring
                    import importlib

                    module_parts = module_path.split(".")
                    module_name = ".".join(module_parts[:-1])
                    func_name = module_parts[-1]

                    actual_module = importlib.import_module(module_name)
                    actual_func = getattr(actual_module, func_name)

                    # If the actual function has a docstring but griffe obj doesn't, use it
                    if actual_func.__doc__ and (
                        not obj.docstring or not obj.docstring.value
                    ):
                        from griffe import Docstring

                        obj.docstring = Docstring(actual_func.__doc__, lineno=1)
                except:
                    pass  # Fall back to griffe's detection

            if obj.docstring:
                # Force parsing with Google parser to get structured sections
                obj.docstring.parsed = griffe.parse_google(obj.docstring)

            # Handle different object types
            if hasattr(obj, "members"):
                # This is a class or module - parse docstrings for all methods/functions
                for member_name, member in obj.members.items():
                    if member.docstring:
                        member.docstring.parsed = griffe.parse_google(member.docstring)

            # Create ConfigDict with the options
            # Adjust default options based on object type
            if hasattr(obj, "kind") and obj.kind.value == "function":
                # Configuration for functions
                default_options = {
                    "docstring_section_style": "table",
                    "heading_level": 3,
                    "show_root_heading": True,
                    "show_source": True,
                    "show_signature": True,
                }
            else:
                # Configuration for classes and modules
                default_options = {
                    "docstring_section_style": "table",
                    "heading_level": 3,
                    "show_root_heading": True,
                    "show_source": True,
                    "summary": {"functions": False},
                }

            default_options.update(options)
            config = ConfigDict(**default_options)

            # Generate the documentation using griffe2md
            # Type ignore since griffe2md can handle various object types
            markdown_docs = render_object_docs(obj, config)  # type: ignore

            markdown_docs = markdown_docs.replace(f"### `{to_replace}.", "### `")

            return markdown_docs

        except Exception as e:
            return f"<!-- Error generating docs for {module_path}: {str(e)} -->"

    def process_markdown(self, content: str) -> str:
        """Process markdown content, replacing ::: blocks with generated documentation"""

        # Pattern to match ::: blocks (including multi-line YAML config)
        pattern = r":::\s*([^\n]+)(?:\n((?:\s{4}.*\n?)*))?"

        def replace_block(match):
            module_line = match.group(1).strip()
            yaml_block = match.group(2) or ""

            # Reconstruct the full block
            full_block = f":::{module_line}\n{yaml_block}".rstrip()

            try:
                module_path, handler_type, options = self.parse_docstring_block(
                    full_block
                )
                generated_docs = self.generate_documentation(module_path, options)
                return generated_docs
            except Exception as e:
                return f"<!-- Error processing block: {str(e)} -->"

        return re.sub(pattern, replace_block, content, flags=re.MULTILINE)

    def process_file(self, input_file: str, output_file: Optional[str] = None) -> str:
        """Process a markdown file and return the result"""
        with open(input_file, "r", encoding="utf-8") as f:
            content = f.read()

        processed_content = self.process_markdown(content)

        if output_file:
            with open(output_file, "w", encoding="utf-8") as f:
                f.write(processed_content)

        return processed_content

    def get_args(self):
        parser = argparse.ArgumentParser(
            description="Convert ::: blocks to mkdocstrings"
        )
        parser.add_argument("input_file", type=str, help="Input markdown file")
        parser.add_argument("output_file", type=str, help="Output markdown file")
        return parser.parse_args()


# Usage example
if __name__ == "__main__":
    parser = MkDocstringsParser()

    # Test with a class
    test_class = """::: coreforecast.lag_transforms.Lag
    handler: python
    options:
      docstring_style: google
      members:
        - stack
        - take
        - transform
        - update
      heading_level: 3
      show_root_heading: true
      show_source: true
"""

    # Test with a function (using a real function)
    test_function = """::: coreforecast.differences.diff
    handler: python
    options:
      docstring_style: google
      heading_level: 3
      show_root_heading: true
      show_source: true
      show_signature: true
"""

    print("Class documentation:")
    print(parser.process_markdown(test_class))
    print("\n" + "=" * 50 + "\n")
    print("Function documentation:")
    fn = parser.process_markdown(test_function)
    print(fn)
    # console = Console()
    # console.print(Markdown(fn))

    # args = parser.get_args()
    # parser.process_file(args.input_file, args.output_file)
