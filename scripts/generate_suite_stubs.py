#!/usr/bin/env python3
"""
Script to generate .pyi stub file for DataFrameExpectationsSuite.

This script copies suite.py and transforms it into a .pyi stub file by:
1. Removing implementation details (function bodies replaced with ...)
2. Keeping type hints, signatures, and docstrings
3. Adding dynamically generated expectation methods from the registry

Usage:
    python scripts/generate_suite_stubs.py          # Generate suite.pyi
    python scripts/generate_suite_stubs.py --check  # Only check if stub file is up-to-date
    python scripts/generate_suite_stubs.py --print  # Print generated stubs to stdout
"""

import argparse
import ast
import sys
from pathlib import Path
from typing import Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def transform_suite_to_stub() -> str:
    """
    Transform suite.py into stub format using AST parsing.

    Returns:
        The stub content as a string
    """
    suite_file = Path(__file__).parent.parent / 'dataframe_expectations' / 'suite.py'

    with open(suite_file, 'r') as f:
        source = f.read()

    # Parse the source code
    tree = ast.parse(source)

    # Transform to stub
    stub_lines = []

    # Process imports
    for node in tree.body:
        match node:
            case ast.ImportFrom(module='dataframe_expectations.logging_utils'):
                # Skip logger import
                continue
            case ast.ImportFrom(module='typing') as import_node:
                # Add Union to typing imports if not present
                unparsed = ast.unparse(import_node)
                if 'Union' not in unparsed:
                    unparsed = unparsed.replace('from typing import ', 'from typing import Union, ')
                stub_lines.append(unparsed)
            case ast.Import() | ast.ImportFrom():
                stub_lines.append(ast.unparse(node))

    stub_lines.append('')  # Empty line after imports

    # Process classes
    for node in tree.body:
        match node:
            case ast.ClassDef():
                stub_lines.append(format_class_stub(node))

    return '\n'.join(stub_lines)


def format_class_stub(class_node: ast.ClassDef) -> str:
    """Format a class as a stub with all methods replaced by ..."""
    lines = []

    # Class definition
    bases = ', '.join(ast.unparse(base) for base in class_node.bases)
    if bases:
        lines.append(f'class {class_node.name}({bases}):')
    else:
        lines.append(f'class {class_node.name}:')

    # Class docstring
    match class_node.body:
        # Check for docstring in first statement
        case [ast.Expr(value=ast.Constant(value=str() as docstring)), *_]:
            lines.append(f'    """{docstring}"""')
            body_start = 1
        case _:
            body_start = 0

    # Process methods and properties
    for item in class_node.body[body_start:]:
        match item:
            case ast.FunctionDef():
                lines.append(format_method_stub(item))
            case ast.Assign():
                # Keep class-level assignments
                lines.append(f'    {ast.unparse(item)}')

    lines.append('')  # Empty line after class
    return '\n'.join(lines)


def format_method_stub(func_node: ast.FunctionDef) -> str:
    """Format a method as a stub."""
    lines = []

    # Handle decorators
    for decorator in func_node.decorator_list:
        lines.append(f'    @{ast.unparse(decorator)}')

    # Method signature
    args = ast.unparse(func_node.args)
    returns = f' -> {ast.unparse(func_node.returns)}' if func_node.returns else ''
    lines.append(f'    def {func_node.name}({args}){returns}:')

    # Method docstring
    match func_node.body:
        # Check for docstring in first statement
        case [ast.Expr(value=ast.Constant(value=str() as docstring)), *_]:
            # Format docstring with proper indentation
            docstring_lines = docstring.split('\n')
            lines.append('        """')
            for line in docstring_lines:
                lines.append(f'        {line}' if line.strip() else '')
            lines.append('        """')

    # Add ...
    lines.append('        ...')

    return '\n'.join(lines)


def format_type_hint(type_hint: Any) -> str:
    """Format a type hint for use in function signatures."""
    if type_hint is None:
        return "object"

    # Handle tuple of types (e.g., (int, float) -> Union[int, float])
    if isinstance(type_hint, tuple):
        type_names = []
        for t in type_hint:
            if hasattr(t, '__name__'):
                type_names.append(t.__name__)
            else:
                type_names.append(str(t))

        if len(type_names) == 1:
            return type_names[0]
        return f"Union[{', '.join(type_names)}]"

    # Handle single type
    if hasattr(type_hint, '__name__'):
        return type_hint.__name__

    return str(type_hint)


def generate_stub_method(
    suite_method_name: str,
    expectation_name: str,
    metadata: Any,
) -> str:
    """Generate a stub method signature for a single expectation."""
    description = metadata.pydoc
    category = metadata.category
    subcategory = metadata.subcategory
    params = metadata.params
    params_doc = metadata.params_doc
    param_types = metadata.param_types

    # Build parameter list with type hints
    param_list = []
    for param in params:
        param_type = param_types.get(param, object)
        type_str = format_type_hint(param_type)
        param_list.append(f"{param}: {type_str}")

    # Add tags parameter (always optional)
    param_list.append("tags: Optional[List[str]] = None")

    params_signature = ",\n        ".join(param_list)

    # Build docstring
    category_str = category.value if hasattr(category, 'value') else str(category)
    subcategory_str = subcategory.value if hasattr(subcategory, 'value') else str(subcategory)

    docstring_lines = [
        '        """',
        f'        {description}',
        '',
        '        Categories:',
        f'          category: {category_str}',
        f'          subcategory: {subcategory_str}',
        '',
    ]

    # Add parameter documentation
    if params:
        for param in params:
            param_doc = params_doc.get(param, '')
            docstring_lines.append(f'        :param {param}: {param_doc}')
        docstring_lines.append('')

    # Add tags parameter documentation
    docstring_lines.append('        :param tags: Optional tags as list of strings in "key:value" format (e.g., ["priority:high", "env:test"]).')
    docstring_lines.append('')

    # Add return documentation
    docstring_lines.append('        :return: An instance of DataFrameExpectationsSuite.')
    docstring_lines.append('        """')

    # Generate the method signature for .pyi file
    method_code = f'''    def {suite_method_name}(
        self,
        {params_signature},
    ) -> DataFrameExpectationsSuite:
{chr(10).join(docstring_lines)}
        ...

'''

    return method_code


def generate_pyi_file() -> str:
    """Generate complete .pyi stub file content."""
    # Import here to avoid issues if not in the right directory
    from dataframe_expectations.registry import (
        DataFrameExpectationRegistry,
    )

    # Start with header
    pyi_content = """\
# Type stubs for DataFrameExpectationsSuite
# Auto-generated by scripts/generate_suite_stubs.py
# DO NOT EDIT - Regenerate with: python scripts/generate_suite_stubs.py

"""

    # Get the base stub from suite.py
    base_stub = transform_suite_to_stub()

    # Find where to insert the dynamic methods (before __getattr__)
    getattr_pos = base_stub.find('    def __getattr__(self, name: str)')

    if getattr_pos == -1:
        raise ValueError("Could not find __getattr__ method in transformed suite")

    # Get all metadata and suite method mapping
    mapping = DataFrameExpectationRegistry.get_suite_method_mapping()

    # Generate all stub methods (sorted for consistency)
    dynamic_methods = '\n'
    for suite_method, exp_name in sorted(mapping.items()):
        metadata = DataFrameExpectationRegistry.get_metadata(exp_name)
        dynamic_methods += generate_stub_method(suite_method, exp_name, metadata)

    # Insert dynamic methods before __getattr__
    pyi_content += base_stub[:getattr_pos] + dynamic_methods + base_stub[getattr_pos:]

    return pyi_content


def update_pyi_file(dry_run: bool = False) -> bool:
    """Update the suite.pyi stub file."""
    pyi_file = Path(__file__).parent.parent / 'dataframe_expectations' / 'suite.pyi'

    # Generate the new .pyi content
    new_content = generate_pyi_file()

    # Check if file exists and compare
    old_content = ''
    if pyi_file.exists():
        with open(pyi_file, 'r') as f:
            old_content = f.read()

    if new_content == old_content:
        print("✅ Stub file is up-to-date")
        return False

    if dry_run:
        print("❌ Stub file is out of date. Run without --check to update.")
        return True

    # Write the .pyi file
    with open(pyi_file, 'w') as f:
        f.write(new_content)

    # Count the methods
    from dataframe_expectations.registry import (
        DataFrameExpectationRegistry,
    )
    method_count = len(DataFrameExpectationRegistry.get_suite_method_mapping())

    print(f"✅ Successfully generated {method_count} stub methods in {pyi_file}")
    return True


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate .pyi stub file for DataFrameExpectationsSuite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate suite.pyi
  python scripts/generate_suite_stubs.py

  # Check if stub file is up-to-date (useful for CI)
  python scripts/generate_suite_stubs.py --check

  # Print generated stubs to stdout
  python scripts/generate_suite_stubs.py --print
        """
    )
    parser.add_argument(
        '--check',
        action='store_true',
        help='Check if stub file is up-to-date without modifying it'
    )
    parser.add_argument(
        '--print',
        action='store_true',
        help='Print generated stubs to stdout instead of writing to file'
    )

    args = parser.parse_args()

    if args.print:
        # Print to stdout
        print(generate_pyi_file())
        return

    # Update or check the file
    was_updated = update_pyi_file(dry_run=args.check)

    # Exit with error code if --check and file is out of date
    if args.check and was_updated:
        sys.exit(1)


if __name__ == '__main__':
    main()
