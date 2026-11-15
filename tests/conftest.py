"""Shared input selection for optional real-evidence figure integration tests."""
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]


def pytest_addoption(parser):
    parser.addoption('--figure-tables', default='recomputed/paper-tables',
                     help='Completed paper-tables output; relative paths are repository-relative')


@pytest.fixture(scope='session')
def paper_tables(pytestconfig):
    root = Path(pytestconfig.getoption('--figure-tables'))
    if not root.is_absolute():
        root = REPO / root
    if not root.exists():
        pytest.skip('Figure integration inputs unavailable: ' + str(root) +
                    '; generate paper-tables and pass --figure-tables PATH')
    # An existing but incomplete or altered export is a failure, not a skip.
    from analysis.revision_paper_figures import validate_tables
    validate_tables(REPO, root)
    return root.resolve()
