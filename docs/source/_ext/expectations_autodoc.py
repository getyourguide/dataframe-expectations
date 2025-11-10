"""
Custom Sphinx extension for generating categorized DataFrame expectations documentation.
"""

import inspect
import re
from collections import defaultdict
from typing import Any, Dict, List, Tuple

from docutils import nodes
from docutils.nodes import Node
from docutils.parsers.rst import directives
from sphinx.application import Sphinx
from sphinx.util.docutils import SphinxDirective

from dataframe_expectations.registry import DataFrameExpectationRegistry
from dataframe_expectations.suite import DataFrameExpectationsSuite


def parse_metadata_from_docstring(docstring: str) -> Tuple[str, str]:
    """Parse metadata from docstring using YAML-style format."""
    if not docstring:
        return None, None

    # Look for Categories section with YAML-style indentation
    pattern = r"Categories:\s*\n\s*category:\s*(.+)\n\s*subcategory:\s*(.+)"
    match = re.search(pattern, docstring, re.IGNORECASE)

    if match:
        return match.group(1).strip(), match.group(2).strip()

    return None, None


def infer_category_from_method_name(method_name: str) -> Tuple[str, str]:
    """Infer category and subcategory from method name as fallback."""
    if any(
        keyword in method_name
        for keyword in ["quantile", "max", "min", "mean", "median", "unique_rows"]
    ):
        return "Column Aggregation Expectations", get_subcategory_from_name(method_name)
    else:
        return "Column Expectations", get_subcategory_from_name(method_name)


def get_subcategory_from_name(method_name: str) -> str:
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


class ExpectationsDirective(SphinxDirective):
    """
    Custom directive to generate categorized expectations documentation.

    Usage:
    .. expectations::
       :class: dataframe_expectations.expectations_suite.DataFrameExpectationsSuite
       :show-summary: true
       :show-cards: true
    """

    has_content = False
    required_arguments = 0
    optional_arguments = 0
    option_spec = {
        'class': directives.unchanged_required,
        'show-summary': directives.flag,
        'show-cards': directives.flag,
    }

    def run(self) -> List[Node]:
        """Generate the expectations documentation."""
        # Get all registered expectations from the registry
        expectations_by_category = defaultdict(lambda: defaultdict(list))
        method_details = {}

        # Get the suite method mapping from the registry (method_name -> expectation_name)
        suite_method_mapping = DataFrameExpectationRegistry.get_suite_method_mapping()

        # Get all metadata from the registry
        for suite_method_name, expectation_name in suite_method_mapping.items():
            metadata = DataFrameExpectationRegistry.get_metadata(expectation_name)

            # Get category and subcategory from metadata
            category = metadata.category.value if hasattr(metadata.category, 'value') else str(metadata.category)
            subcategory = metadata.subcategory.value if hasattr(metadata.subcategory, 'value') else str(metadata.subcategory)

            expectations_by_category[category][subcategory].append(suite_method_name)

            # Build docstring from metadata
            docstring_lines = [
                metadata.pydoc,
                "",
                "Categories:",
                f"  category: {category}",
                f"  subcategory: {subcategory}",
                ""
            ]

            # Add parameters documentation
            if metadata.params:
                for param in metadata.params:
                    param_doc = metadata.params_doc.get(param, "")
                    docstring_lines.append(f":param {param}: {param_doc}")
                docstring_lines.append("")

            docstring_lines.append(":return: An instance of DataFrameExpectationsSuite")
            docstring = "\n".join(docstring_lines)

            # Build method signature
            param_strs = []
            for param in metadata.params:
                param_type = metadata.param_types.get(param, object)
                type_name = param_type.__name__ if hasattr(param_type, '__name__') else str(param_type)
                param_strs.append(f"{param}: {type_name}")

            method_details[suite_method_name] = {
                "method": None,  # No actual method object since it's dynamic
                "docstring": docstring,
                "signature_str": f"({', '.join(param_strs)})" if param_strs else "()",
                "params": metadata.params,
                "category": category,
                "subcategory": subcategory,
            }

        # Generate nodes
        nodes_list = []

        # Add summary table if requested
        if 'show-summary' in self.options:
            nodes_list.extend(self._generate_summary_table(expectations_by_category, method_details))

        # Add cards if requested
        if 'show-cards' in self.options:
            nodes_list.extend(self._generate_expectation_cards(expectations_by_category, method_details))

        return nodes_list

    def _generate_summary_table(self, expectations_by_category, method_details) -> List[Node]:
        """Generate summary table nodes."""
        nodes_list = []

        # Add section with title and proper ID
        summary_section = nodes.section()
        summary_section['ids'] = ['expectations-summary']
        summary_section['names'] = ['expectations-summary']
        summary_title = nodes.title("", "Expectations Summary")
        summary_section += summary_title

        # Create table
        table = nodes.table()
        tgroup = nodes.tgroup(cols=3)
        table += tgroup

        # Add column specifications
        for width in [30, 25, 45]:
            colspec = nodes.colspec(colwidth=width)
            tgroup += colspec

        # Add table head
        thead = nodes.thead()
        tgroup += thead

        row = nodes.row()
        thead += row

        for header in ["Category", "Subcategory", "Expectations"]:
            entry = nodes.entry()
            row += entry
            entry += nodes.paragraph("", header)

        # Add table body
        tbody = nodes.tbody()
        tgroup += tbody

        for category in sorted(expectations_by_category.keys()):
            for subcategory in sorted(expectations_by_category[category].keys()):
                expectations = expectations_by_category[category][subcategory]

                row = nodes.row()
                tbody += row

                # Category cell
                entry = nodes.entry()
                row += entry
                entry += nodes.paragraph("", category)

                # Subcategory cell
                entry = nodes.entry()
                row += entry
                entry += nodes.paragraph("", subcategory)

                # Expectations cell
                entry = nodes.entry()
                row += entry

                exp_para = nodes.paragraph()
                for i, exp in enumerate(sorted(expectations)):
                    if i > 0:
                        exp_para += nodes.Text(", ")

                    # Create clickable link to the card using raw HTML
                    raw_link = nodes.raw(
                        f'<a href="#card-{exp}" class="expectation-link">{exp}</a>',
                        f'<a href="#card-{exp}" class="expectation-link">{exp}</a>',
                        format='html'
                    )
                    exp_para += raw_link

                entry += exp_para

        summary_section += table
        nodes_list.append(summary_section)
        return nodes_list

    def _generate_expectation_cards(self, expectations_by_category, method_details) -> List[Node]:
        """Generate expectation cards in Great Expectations gallery style."""
        nodes_list = []

        for category in sorted(expectations_by_category.keys()):
            # Category header - use proper heading for TOC inclusion as top-level section
            cat_section = nodes.section()
            cat_section['ids'] = [f"category-{category.lower().replace(' ', '-')}"]
            cat_section['names'] = [category.lower().replace(' ', '-')]

            cat_header = nodes.title("", category)
            cat_header['classes'] = ['category-title']
            cat_section += cat_header

            # Create cards container for this category
            cards_container = nodes.container()
            cards_container['classes'] = ['expectations-gallery']

            for subcategory in sorted(expectations_by_category[category].keys()):
                # Subcategory header - use paragraph with special styling
                subcat_header = nodes.paragraph()
                subcat_header['classes'] = ['subcategory-title']
                subcat_header += nodes.Text(subcategory)
                cards_container += subcat_header

                # Cards grid for this subcategory
                cards_grid = nodes.container()
                cards_grid['classes'] = ['cards-grid']

                for method_name in sorted(expectations_by_category[category][subcategory]):
                    details = method_details[method_name]
                    card = self._create_expectation_card(method_name, details)
                    cards_grid += card

                cards_container += cards_grid

            cat_section += cards_container
            nodes_list.append(cat_section)

        return nodes_list

    def _create_expectation_card(self, method_name: str, details: dict) -> Node:
        """Create a single expectation card."""
        # Create card container
        card = nodes.container()
        card['classes'] = ['expectation-card']
        card['ids'] = [f"card-{method_name}"]

        # Card header with method name
        card_header = nodes.container()
        card_header['classes'] = ['card-header']

        method_title = nodes.paragraph()
        method_title['classes'] = ['method-name']
        method_title += nodes.Text(method_name)
        card_header += method_title

        card += card_header

        # Card body
        card_body = nodes.container()
        card_body['classes'] = ['card-body']

        # Description
        clean_docstring = clean_docstring_from_metadata(details["docstring"])
        if clean_docstring:
            description = clean_docstring.split('\n')[0]  # First line only
            desc_para = nodes.paragraph()
            desc_para['classes'] = ['card-description']
            desc_para += nodes.Text(description)
            card_body += desc_para

        # Data quality issue tags (similar to Great Expectations)
        tags_container = nodes.container()
        tags_container['classes'] = ['tags-container']

        # Add category as a tag
        category_tag = nodes.inline()
        category_tag['classes'] = ['tag', 'category-tag']
        category_tag += nodes.Text(details['category'])
        tags_container += category_tag

        # Add subcategory as a tag
        subcategory_tag = nodes.inline()
        subcategory_tag['classes'] = ['tag', 'subcategory-tag']
        subcategory_tag += nodes.Text(details['subcategory'])
        tags_container += subcategory_tag

        card_body += tags_container

        # Parameters preview
        params = details.get("params", [])
        if params:
            params_container = nodes.container()
            params_container['classes'] = ['params-preview']

            params_title = nodes.paragraph()
            params_title['classes'] = ['params-title']
            params_title += nodes.Text("Parameters:")
            params_container += params_title

            params_list = nodes.paragraph()
            params_list['classes'] = ['params-list']
            params_text = ", ".join(params[:3])  # Show first 3 parameters
            if len(params) > 3:
                params_text += f", ... (+{len(params) - 3} more)"
            params_list += nodes.Text(params_text)
            params_container += params_list

            card_body += params_container

        card += card_body

        return card


def setup(app: Sphinx) -> Dict[str, Any]:
    """Setup function for the Sphinx extension."""
    app.add_directive("expectations", ExpectationsDirective)

    return {
        'version': '0.1',
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }
