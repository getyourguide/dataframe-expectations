import os
import sys

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

# Add the project root and extension directories to the path
sys.path.insert(0, os.path.abspath('../../'))
sys.path.insert(0, os.path.abspath('_ext'))

# Read version from pyproject.toml
with open(os.path.abspath('../../pyproject.toml'), 'rb') as f:
    pyproject = tomllib.load(f)
    release = pyproject['project']['version']

# Project information
project = 'DataFrame Expectations'
copyright = '2025, GetYourGuide'
author = 'Data Products GetYourGuide'

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
html_theme = 'sphinx_book_theme'

# Sphinx Book Theme options
html_theme_options = {
    "repository_url": "https://github.com/getyourguide/dataframe-expectations",
    "use_repository_button": True,
    "use_edit_button": False,
    "use_issues_button": True,
    "use_download_button": True,
    "path_to_docs": "docs/source",
    "show_navbar_depth": 2,
    "show_toc_level": 2,
    "home_page_in_toc": True,
    "navigation_with_keys": True,
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
html_title = f"{project} v{release}"
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
