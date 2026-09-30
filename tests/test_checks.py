"""The real pipeline passes every check on the evaluation harness."""

import pytest

from gil.pipeline import REAL
from validation.checks import CHECKS


@pytest.mark.parametrize("check", CHECKS, ids=[c.name for c in CHECKS])
def test_real_pipeline_passes(check):
    check.run(REAL)


def test_every_check_has_a_known_kind():
    kinds = {"reference", "invariant", "leakage", "sanity", "metamorphic", "determinism"}
    assert {c.kind for c in CHECKS} <= kinds
    assert len({c.name for c in CHECKS}) == len(CHECKS)
