# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys

sys.path.insert(0, os.path.abspath("../.."))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Utilities"
copyright = "2025, Kim Neuroscience Lab"
author = "Kim Neuroscience Lab"
release = "1.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.githubpages",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
    "myst_parser",
]

templates_path = ["_templates"]
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
html_title = "Kim Lab Utilities"
html_logo = "_static/logo.png"  # You can add your lab's logo here
html_favicon = "_static/favicon.ico"  # You can add your lab's favicon here

# Theme options
html_theme_options = {
    "display_version": True,
    "prev_next_buttons_location": "bottom",
    "style_external_links": False,
    "style_nav_header_background": "#2980B9",
    # Toc options
    "collapse_navigation": True,
    "sticky_navigation": True,
    "navigation_depth": 4,
    "includehidden": True,
    "titles_only": False,
}

# -- Extension configuration -------------------------------------------------
autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_typehints_format = "short"

# Napoleon settings
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = True
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = True
napoleon_use_ivar = True
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_type_aliases = None

# Intersphinx settings
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "matplotlib": ("https://matplotlib.org/stable/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "torch": ("https://pytorch.org/docs/stable/", None),
}
