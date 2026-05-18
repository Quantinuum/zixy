# Copyright 2026 Quantinuum
"""Configuration file for the Sphinx documentation builder."""

# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

from datetime import datetime
from typing import Any

# Alias needed here to prevent naming clash with something else sphinx is doing.
from importlib.metadata import version as check_version

# Checking the version used in the Python environment means that pyproject.toml
#  serves as the single source of truth for the version of zixy.


project = f"zixy-py v{check_version('zixy')}"
copyright = f"{datetime.now().year}, Quantinuum"
author = "Quantinuum"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_nb",
    "sphinx.ext.autosummary",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx.ext.intersphinx",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
]

suppress_warnings = [
    "sphinx_autodoc_typehints.guarded_import",  # suppress warnings about typecheck-time imports
]

templates_path = ["_templates"]

napoleon_google_docstring = True
napoleon_use_param = False  # don't show parameter types in description (they're in the signature)
napoleon_use_rtype = False  # don't show return types in description (they're in the signature)

typehints_use_signature = True  # show parameter types in signature
typehints_use_signature_return = True  # show return type in signature
typehints_document_rtype = False
typehints_fully_qualified = False

autosummary_generate = True
autosummary_ignore_module_all = False  # Respect __all__ if specified

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"

# --- MyST-NB config ---
# https://myst-nb.readthedocs.io/en/latest/configuration.html
nb_execution_mode = "off"
# --------------------------


exclude_patterns = [
    "_build",
    "build/**",
    "**.ipynb_checkpoints",
    "**.pyc",
    "**.py",
    ".venv",
    ".env",
    "README.md",
    "**/README.md",
    ".jupyter_cache",
    "jupyter_execute",
]

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sympy": ("https://docs.sympy.org/latest/", None),
}
