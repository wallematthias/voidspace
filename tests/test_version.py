from pathlib import Path
import tomllib

import voidspace


def test_public_version_matches_package_version():
    project = tomllib.loads(
        (Path(__file__).resolve().parents[1] / "pyproject.toml").read_text()
    )["project"]
    assert voidspace.__version__ == project["version"]
