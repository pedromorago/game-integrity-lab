"""Every seeded bug in the evaluation pipeline is caught by at least one check."""

import pytest

from scripts.run_seeded_bugs import failures
from validation.mutants import MUTANTS


@pytest.mark.slow
@pytest.mark.parametrize("pipeline", [m for m, _ in MUTANTS], ids=[m.name for m, _ in MUTANTS])
def test_seeded_bug_is_caught(pipeline):
    assert failures(pipeline), f"no check caught: {pipeline.name}"


def test_mutant_names_are_unique():
    names = [m.name for m, _ in MUTANTS]
    assert len(set(names)) == len(names)
