from pathlib import Path


def test_closure_source_excludes_operator_wrong_aliases():
    source = Path("src/eeg_cgdr/experiments/raw_support_closure.py").read_text()
    assert 'name.startswith("DIFF-CLEAN-WRONG-")' in source
    assert "donor.isdigit()" in source


def test_closure_labels_and_resume_boundary_are_explicit():
    source = Path("src/eeg_cgdr/experiments/raw_support_closure.py").read_text()
    for label in ("MATCH_OVER_STRONG_POP_NOT_ESTABLISHED", "DONOR_SPECIFICITY_SUGGESTIVE", "ABSOLUTE_NATURAL_SAFETY_NOT_ESTABLISHED"):
        assert label in source
    assert "interrupted-training resume equality was not actually executed" in source
