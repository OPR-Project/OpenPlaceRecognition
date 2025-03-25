# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "opr"
author = "Alexander Melekhin, Vitaly Bezuglyj, Ilia Petryashin, Sergey Linok, Kirill Muravyev, Dmitry Yudin"
copyright = f"2024, {author}"

# -- Internationalization options --------------------------------------------
locale_dirs = ['locale/']
gettext_compact = False
language = 'en'            # default language

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

autodoc_mock_imports = [
    'hregnet',
    'geotransformer',
    'faiss',
    'MinkowskiEngine',
    'torchvision',
    'onnxruntime',
    'paddleocr',
    'torch_tensorrt',
    'polygraphy',
    'skimage',
]

extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',  # for Google style docstrings
    'sphinx.ext.viewcode',  # to add links to source code
]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    'navigation_depth': 4,  # Ensures deeper levels of the TOC are displayed
    'titles_only': False,   # Shows the full TOC tree, not just section titles
    'collapse_navigation': False,
    'sticky_navigation': True,
    'style_external_links': True,
}

# Force the html_theme_path to ensure our custom templates are found
html_theme_path = ["."]

# Add any paths that contain custom static files
html_static_path = ["_static"]

# HTML context settings for GitHub repository link
html_context = {
    "display_github": True,
    "github_user": "OPR-Project",
    "github_repo": "OpenPlaceRecognition",
    "github_version": "main",
    "conf_py_path": "/docs/source/"
}
