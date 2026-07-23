import os
import sys

# Add project root so Sphinx can import the package
sys.path.insert(0, os.path.abspath('..'))

project = 'ScintSuite'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'sphinx.ext.autosummary',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']

# Autodoc options
autodoc_member_order = 'bysource'
autodoc_typehints = 'description'

# Autosummary: generate stub pages for modules/classes/functions
autosummary_generate = True

copyright = '2026, Jose Rueda, Pablo Oyola, Lina Velarde, Javier Hidalgo'
author = 'Jose Rueda, Pablo Oyola, Lina Velarde, Javier Hidalgo'
release = '1.4.3'
