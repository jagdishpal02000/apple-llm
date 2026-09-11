import json
import pathlib
import sys

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = pathlib.Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC))

#: The corpus is shared with the Node package on purpose: both suites read these
#: same files, which is what keeps the two ports honest.
FIXTURE_DIR = REPO_ROOT / "tests" / "fixtures" / "schema"
IMAGE_DIR = REPO_ROOT / "tests" / "fixtures" / "images"


def load_fixtures():
    return [json.loads(p.read_text()) for p in sorted(FIXTURE_DIR.glob("*.json"))]


@pytest.fixture(scope="session")
def repo_root():
    return REPO_ROOT
