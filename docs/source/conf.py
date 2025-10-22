import os
import sys

# Add the project root and extension directories to the path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('_ext'))

# Project information
project = 'DataFrame Expectations'
copyright = '2024, Your Name'
author = 'Your Name'
release = '0.1.0'

# Extensions
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',  # For Google/NumPy style docstrings
    'sphinx.ext.intersphinx',
    'expectations_autodoc',  # Our custom extension
]

# Theme
html_theme = 'pydata_sphinx_theme'

# PyData theme options for modern, full-width usage
html_theme_options = {
    "use_edit_page_button": False,
    "navigation_depth": 3,
    "show_prev_next": True,
    "navbar_persistent": ["search-button"],
    "navbar_center": ["navbar-nav"],
    "navbar_end": [],
    "sidebar_includehidden": True,
    "primary_sidebar_end": ["page-toc"],
    "secondary_sidebar_items": [],
    "show_toc_level": 3,
}

# Autodoc settings
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'show-inheritance': True,
    'special-members': '__init__',
}

# Napoleon settings for docstring parsing
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'pandas': ('https://pandas.pydata.org/pandas-docs/stable/', None),
    'pyspark': ('https://spark.apache.org/docs/latest/api/python/', None),
}

# HTML output options
html_static_path = ['_static']
html_css_files = [
    'custom.css',
]

# Configure HTML title and layout
html_title = f"{project} v{release} Documentation"
html_short_title = project

# PyData theme context
html_context = {
    'display_github': True,
    'github_user': 'getyourguide',
    'github_repo': 'dataframe-expectations',
    'github_version': 'main',
    'doc_path': 'docs/source/',
}

# Logo configuration
html_logo = None  # You can add a logo path here if needed
html_favicon = None  # You can add a favicon path here if needed
