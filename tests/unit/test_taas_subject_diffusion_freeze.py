from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def test_manuscript_scope_and_deployment_language() -> None:
    text = "\n".join(p.read_text() for p in (ROOT / "taas_submission").rglob("*.tex"))
    assert "EOG-guided" in text
    assert "query eog is an explicit deployment input" in text.lower()
    assert "independent replication" not in text.lower()
    assert "state-of-the-art" not in text.lower()
    assert "support-calibrated artifact-subspace" not in text.lower()


def test_frozen_numbers_and_comparison_boundary_present() -> None:
    text = (ROOT / "taas_submission/main.tex").read_text() + (ROOT / "taas_submission/sections/experiments.tex").read_text()
    for value in ("0.03473", "0.04836", "0.11350", "0.11225", "0.08760", "0.001953", "0.003906"):
        assert value in text
    assert "not supported" in text


def test_readme_archives_legacy_headline() -> None:
    readme=(ROOT/"README.md").read_text()
    assert readme.startswith("# Subject-Calibrated EOG-Guided Diffusion")
    assert "Historical SADDPM" in readme
