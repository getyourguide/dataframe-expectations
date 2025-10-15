import inspect
import re
from collections import defaultdict

from dataframe_expectations.expectations_suite import DataframeExpectationsSuite
from dataframe_expectations.logging_utils import setup_logger

logger = setup_logger(__name__)


def parse_metadata_from_docstring(docstring: str):
    """Parse metadata from docstring using YAML-style format."""
    if not docstring:
        return None, None

    # Look for Categories section with YAML-style indentation
    pattern = r"Categories:\s*\n\s*category:\s*(.+)\n\s*subcategory:\s*(.+)"
    match = re.search(pattern, docstring, re.IGNORECASE)

    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None, None


def infer_category_from_method_name(method_name: str):
    """Infer category and subcategory from method name as fallback."""
    if any(
        keyword in method_name
        for keyword in ["quantile", "max", "min", "mean", "median", "unique_rows"]
    ):
        return "Column Aggregation Expectations", get_subcategory_from_name(method_name)
    else:
        return "Column Expectations", get_subcategory_from_name(method_name)


def get_subcategory_from_name(method_name: str):
    """Get subcategory from method name."""
    if any(
        keyword in method_name
        for keyword in ["string", "length", "contains", "starts", "ends"]
    ):
        return "String"
    elif any(
        keyword in method_name
        for keyword in [
            "greater",
            "less",
            "between",
            "quantile",
            "max",
            "min",
            "mean",
            "median",
        ]
    ):
        return "Numerical"
    else:
        return "Any Value"


def clean_docstring_from_metadata(docstring: str) -> str:
    """Remove metadata section from docstring."""
    if not docstring:
        return ""

    # Remove Categories section
    pattern = r"Categories:\s*\n\s*category:.*\n\s*subcategory:.*\n?"
    cleaned = re.sub(pattern, "", docstring, flags=re.IGNORECASE)

    return cleaned.strip()


def generate_github_anchor(text: str) -> str:
    """Generate GitHub-compatible anchor from text.

    GitHub's anchor generation rules for: ##### 🎯 `method_name`
    - Emoji 🎯 becomes -
    - Backticks are removed
    - Spaces become hyphens
    - Underscores are preserved
    - Result: -method_name
    """
    anchor = text.lower()

    # Replace emoji with dash
    anchor = re.sub(r"🎯", "-", anchor)

    # Remove backticks and other markdown syntax
    anchor = re.sub(r"[`*]", "", anchor)

    # Replace spaces with hyphens
    anchor = re.sub(r"\s+", "-", anchor)

    # Remove other special characters but keep alphanumeric, hyphens, and underscores
    anchor = re.sub(r"[^a-zA-Z0-9\-_]", "", anchor)

    # Clean up multiple consecutive hyphens
    anchor = re.sub(r"-+", "-", anchor)

    # Don't strip leading dash - it's part of the anchor from the emoji
    anchor = anchor.rstrip("-")

    return anchor


def generate_simple_github_anchor(text: str) -> str:
    """Generate GitHub-compatible anchor for regular headings (categories/subcategories)."""
    anchor = text.lower()

    # Replace spaces with hyphens
    anchor = re.sub(r"\s+", "-", anchor)

    # Remove special characters but keep alphanumeric, hyphens, and underscores
    anchor = re.sub(r"[^a-zA-Z0-9\-_]", "", anchor)

    # Clean up multiple consecutive hyphens
    anchor = re.sub(r"-+", "-", anchor)

    # Remove leading/trailing hyphens
    anchor = anchor.strip("-")

    return anchor


def generate_markdown_doc(cls, output_file: str):
    """
    Generate Markdown documentation for a class with categorized table.
    """
    # Collect all expectations by category
    expectations_by_category: defaultdict = defaultdict(lambda: defaultdict(list))
    method_details = {}

    # Iterate over all methods in the class
    for name, method in inspect.getmembers(cls, predicate=inspect.isfunction):
        # Skip private methods and methods not starting with "expect_"
        if name.startswith("_") or not name.startswith("expect_"):
            continue

        # Extract metadata
        docstring = inspect.getdoc(method) or "No description provided."

        # Parse from docstring
        category, subcategory = parse_metadata_from_docstring(docstring)
        if not category:
            # Fallback to inference
            category, subcategory = infer_category_from_method_name(name)

        expectations_by_category[category][subcategory].append(name)
        method_details[name] = {
            "method": method,
            "docstring": docstring,
            "signature": inspect.signature(method),
            "category": category,
            "subcategory": subcategory,
        }

    # Track heading occurrences for GitHub's automatic numbering
    heading_counts: defaultdict[str, int] = defaultdict(int)

    # Function to get anchor with numbering
    def get_anchor_with_numbering(heading_text: str) -> str:
        base_anchor = generate_simple_github_anchor(heading_text)
        heading_counts[base_anchor] += 1

        if heading_counts[base_anchor] == 1:
            return base_anchor
        else:
            return f"{base_anchor}-{heading_counts[base_anchor] - 1}"

    # Pre-calculate all anchors by going through the structure in the same order
    category_anchors = {}
    subcategory_anchors = {}

    for category in sorted(expectations_by_category.keys()):
        category_anchors[category] = get_anchor_with_numbering(category)
        for subcategory in sorted(expectations_by_category[category].keys()):
            subcategory_key = f"{category}::{subcategory}"
            subcategory_anchors[subcategory_key] = get_anchor_with_numbering(
                subcategory
            )

    # Start building markdown
    markdown = "# DataFrame Expectations Documentation\n\n"

    # Generate summary table
    markdown += "## Expectations Summary\n\n"
    markdown += "| Category | Subcategory | Expectations |\n"
    markdown += "|----------|-------------|-------------|\n"

    for category in sorted(expectations_by_category.keys()):
        for subcategory in sorted(expectations_by_category[category].keys()):
            expectations = expectations_by_category[category][subcategory]

            # Generate category link
            category_link = f"[{category}](#{category_anchors[category]})"

            # Generate subcategory link
            subcategory_key = f"{category}::{subcategory}"
            subcategory_link = (
                f"[{subcategory}](#{subcategory_anchors[subcategory_key]})"
            )

            # Generate expectation links
            expectations_links = [
                f"[{exp}](#{generate_github_anchor(f'🎯 `{exp}`')})"
                for exp in sorted(expectations)
            ]
            expectations_str = ", ".join(expectations_links)

            markdown += (
                f"| {category_link} | {subcategory_link} | {expectations_str} |\n"
            )

    markdown += "\n---\n\n"

    # Generate detailed documentation
    markdown += "## Detailed Documentation\n\n"

    for category in sorted(expectations_by_category.keys()):
        markdown += f"### {category}\n\n"

        for subcategory in sorted(expectations_by_category[category].keys()):
            markdown += f"#### {subcategory}\n\n"

            for method_name in sorted(expectations_by_category[category][subcategory]):
                details = method_details[method_name]

                # Add method name with proper anchor
                markdown += f"##### 🎯 `{method_name}`\n\n"

                # Add method description (clean it from metadata)
                clean_docstring = clean_docstring_from_metadata(details["docstring"])
                description_lines = clean_docstring.split("\n")
                description = (
                    description_lines[0]
                    if description_lines
                    else "No description provided."
                )
                markdown += f"**Description**: {description.strip()}\n\n"

                # Add method parameters
                markdown += "**Parameters**:\n\n"
                for param_name, param in details["signature"].parameters.items():
                    if param_name == "self":
                        continue
                    param_type = (
                        param.annotation
                        if param.annotation != inspect.Parameter.empty
                        else "No type specified"
                    )
                    markdown += f"- `{param_name}`: {param_type}\n"

                # Add method signature
                markdown += "\n**Signature**:\n\n"
                markdown += f"```python\n{method_name}{details['signature']}\n```\n\n"
                markdown += "---\n\n"

    # Write to file
    with open(output_file, "w") as f:
        f.write(markdown)

    logger.info(f"Markdown documentation generated: {output_file}")

    # Print summary to console
    total_expectations = sum(
        len(subcats)
        for subcats in expectations_by_category.values()
        for subcats in subcats.values()
    )
    logger.info(
        f"Generated documentation for {total_expectations} expectations across {len(expectations_by_category)} categories"
    )


# Generate documentation for the DataframeExpectationsSuite class
if __name__ == "__main__":
    generate_markdown_doc(
        DataframeExpectationsSuite,
        output_file="mltools/dataframe_expectations/docs/expectations.md",
    )
