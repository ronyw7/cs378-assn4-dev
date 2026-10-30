"""Verify that the supplied reference reproduces the saved Vidur job latencys."""

import pytest
from conftest import REPO

from agentkv import score


@pytest.mark.parametrize("split", ["public_dev", "public_val"])
def test_reference_matches_the_frozen_results(split):
    r = score(str(REPO / "policies" / "reference.py"), split=split, workers=2)
    assert r.valid
    assert all(t.ratio == 1.0 for t in r.by_trace.values())
