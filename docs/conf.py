from importlib.metadata import version as distribution_version

project = "EnSurv"
release = distribution_version("ensurv")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.mathjax",
    "sphinx_copybutton",
    "sphinx_design",
]

exclude_patterns = ["_build"]
autodoc_member_order = "bysource"
autodoc_typehints = "description"

html_theme = "pydata_sphinx_theme"
html_title = "EnSurv documentation"
templates_path = ["_templates"]
html_static_path = ["_static"]
html_css_files = ["ensurv.css"]
html_sidebars = {"index": []}
html_theme_options = {
    "navbar_align": "left",
    "logo": {"text": "EnSurv"},
    "show_prev_next": True,
    "footer_start": ["author-credit"],
    "footer_end": [],
    "secondary_sidebar_items": ["page-toc"],
}
