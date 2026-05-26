from typer.testing import CliRunner

from waferlens.cli import app


def test_help():
    r = CliRunner().invoke(app, ["--help"])
    assert r.exit_code == 0
    assert "train" in r.stdout
    assert "triage" in r.stdout


def test_make_synthetic(tmp_path):
    out = tmp_path / "s.npz"
    r = CliRunner().invoke(app, ["make-synthetic", "--n", "30", "--out", str(out)])
    assert r.exit_code == 0
    assert out.exists()
