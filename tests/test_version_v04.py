"""Prevent confusing public release metadata with the imported package version."""
from pathlib import Path
import tomllib
import omega_geodepth

def test_version_matches_pyproject():
    root=Path(__file__).resolve().parents[1]
    expected=tomllib.loads((root/'pyproject.toml').read_text())['project']['version']
    assert omega_geodepth.__version__==expected
