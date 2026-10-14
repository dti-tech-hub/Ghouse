"""Dashboard pages: static guards against silent Streamlit runtime breakage.

A Streamlit page is executed as a script, so nothing at import time tells you
that a widget call will raise. Two bugs shipped this way before this file
existed:

* a page used ``st.columns`` without ``import streamlit as st``
  -> ``NameError: name 'st' is not defined`` on every page;
* a page used ``st.page_link("pages/<file>.py", ...)`` -> Streamlit >= 1.36
  raises ``StreamlitPageNotFoundError`` unless the page was registered with
  ``st.Page`` and passed to ``st.navigation``.

These tests are static on purpose: they need no backend, no browser and no
network, so they can run in CI. ``scripts/check_dashboard.py`` performs the
full end-to-end check by executing every page against a live API.
"""

from __future__ import annotations

import ast
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
DASHBOARD = ROOT / "dashboard"
PAGES_DIR = DASHBOARD / "pages"
ENTRYPOINT = DASHBOARD / "app.py"
PAGE_FILES = sorted(PAGES_DIR.glob("*.py"))

REASONS = (
    "1_System_Overview.py",
    "2_Knowledge_Base.py",
    "3_Query_Playground.py",
    "4_Agent_Trace.py",
    "5_Failure_Detection.py",
    "6_Diagnosis.py",
    "7_Recovery.py",
    "8_Experiments.py",
    "9_Metrics.py",
    "10_Fault_Injection.py",
    "11_Model_Comparison.py",
    "12_Export.py",
)


def _source(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _imports_streamlit(path: Path) -> bool:
    try:
        tree = ast.parse(_source(path))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Import)
        and any(alias.name == "streamlit" for alias in node.names)
        for node in ast.walk(tree)
    )


def _uses_st(path: Path) -> bool:
    try:
        tree = ast.parse(_source(path))
    except SyntaxError:
        return False
    return any(
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "st"
        for node in ast.walk(tree)
    )


def test_all_twelve_pages_exist():
    assert [p.name for p in PAGE_FILES] == sorted(REASONS)


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.stem)
def test_every_page_compiles(page: Path):
    compile(_source(page), str(page), "exec")


@pytest.mark.parametrize("page", PAGE_FILES, ids=lambda p: p.stem)
def test_every_page_imports_streamlit(page: Path):
    """Regression: pages rendered st.* without importing the module."""
    assert _imports_streamlit(page), (
        f"{page.name} references st.* but does not `import streamlit as st`; "
        "the page will raise NameError when Streamlit executes it"
    )


@pytest.mark.parametrize("module", [ENTRYPOINT, DASHBOARD / "common.py", *PAGE_FILES], ids=lambda p: p.stem)
def test_modules_using_st_import_it(module: Path):
    assert _imports_streamlit(module), f"{module.name} uses st.* without importing streamlit"


def test_page_link_uses_registered_navigation():
    """Regression: bare page_link paths raise StreamlitPageNotFoundError."""
    for page in PAGE_FILES:
        source = _source(page)
        assert "st.page_link(\"pages/" not in source, (
            f"{page.name} links to a page by path; register pages with "
            "st.Page + st.navigation in dashboard/app.py instead"
        )
    assert "st.navigation" in _source(ENTRYPOINT)
    assert "st.Page(" in _source(DASHBOARD / "common.py")


def test_widget_keys_are_unique_across_entrypoint_and_sidebar():
    """Regression: two widgets sharing a key raise DuplicateWidgetID.

    st.navigation().run() executes the selected page *and then returns*, so a
    widget rendered by app.py after it would collide with the same widget the
    page renders in its sidebar.
    """
    entrypoint = _source(ENTRYPOINT)
    assert entrypoint.count('key="base_url"') == 0, (
        "app.py must not render the base_url input; common.sidebar() owns it"
    )
    assert "text_input(\"API base URL\", key=\"base_url\")" in _source(DASHBOARD / "common.py")


def test_page_registry_matches_files_on_disk():
    from dashboard.api_client import PAGES

    registered = sorted(f"{page_file}.py" for page_file, _ in PAGES)
    assert registered == sorted(REASONS)
    for page_file, _ in PAGES:
        assert (PAGES_DIR / f"{page_file}.py").is_file(), f"missing page file for {page_file}"


def test_every_page_defines_a_main_entry_point():
    """A Streamlit page script is executed top to bottom, so it calls main()
    directly rather than guarding it with ``if __name__ == '__main__'``."""
    for page in PAGE_FILES:
        source = _source(page)
        assert "def main(" in source, f"{page.name} has no main() to register"
        assert 'if __name__ == "__main__":' not in source, (
            f"{page.name} guards main() behind __main__; a Streamlit page is run "
            "as a script and would render nothing"
        )
        tree = ast.parse(source)
        calls_main = any(
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Call)
            and isinstance(node.value.func, ast.Name)
            and node.value.func.id == "main"
            for node in tree.body
        )
        assert calls_main, f"{page.name} never calls main() at module level"


# ---------------------------------------------------------------------------
# mixed-type payloads (ArrowInvalid in st.dataframe)
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("<= 0", "<= 0"),
        (0.62, "0.62"),
        (3, "3"),
        (True, "true"),
        (False, "false"),
        (None, "-"),
        ([60, 3000], "[60, 3000]"),
        ({"a": 1, "b": False}, "{a: 1, b: false}"),
        ([True, 2], "[true, 2]"),
    ],
)
def test_scalar_text_coerces_every_verification_shape(value, expected):
    from dashboard.common import scalar_text

    assert scalar_text(value) == expected


def test_scalar_text_makes_criteria_homogeneous():
    """Regression: mixed dtypes in one column made pyarrow raise ArrowInvalid."""
    from dashboard.common import scalar_text

    criteria = [
        {"criterion": "a", "observed": 0.83, "required": 0.62},
        {"criterion": "b", "observed": 3, "required": "<= 0"},
        {"criterion": "c", "observed": 1840, "required": "[60, 3000]"},
        {"criterion": "d", "observed": False, "required": False},
    ]
    rows = [{k: scalar_text(v) for k, v in record.items()} for record in criteria]
    assert all(isinstance(value, str) for row in rows for value in row.values())


@pytest.mark.parametrize("nested", ["recoveries", "verification", "spans"])
def test_nested_payloads_are_not_passed_straight_to_dataframe(nested):
    """Nested dicts/lists must go through common.table(), which coerces cells."""
    for page in PAGE_FILES:
        for line in _source(page).splitlines():
            if "st.dataframe" in line and nested in line:
                pytest.fail(
                    f"{page.name} passes the nested '{nested}' payload to "
                    "st.dataframe directly; use common.table() so cells are coerced"
                )


def test_no_undefined_names_in_dashboard_modules():
    """Regression: a page used `table` it had not imported -> NameError.

    A Streamlit page is only executed when a user opens it, so an undefined
    name hides until then. ruff's F821 check is the static equivalent.
    """
    ruff = shutil.which("ruff")
    command = [ruff] if ruff else [sys.executable, "-m", "ruff"]
    if ruff is None:
        try:
            subprocess.run(command + ["--version"], capture_output=True, check=True)
        except (OSError, subprocess.CalledProcessError):
            pytest.skip("ruff is not installed")
    targets = [str(ENTRYPOINT), str(DASHBOARD / "common.py"), *(str(p) for p in PAGE_FILES)]
    result = subprocess.run(
        [*command, "check", "--select", "F821,F811,F841", "--no-cache", *targets],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "undefined or redefined names in the dashboard:\n" + result.stdout
    )
