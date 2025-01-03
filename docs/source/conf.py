# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "opr"
author = "Alexander Melekhin, Vitaly Bezuglyj, Ilia Petryashin, Sergey Linok, Kirill Muravyev, Dmitry Yudin"
copyright = f"2024, {author}"

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
html_static_path = ["_static"]
