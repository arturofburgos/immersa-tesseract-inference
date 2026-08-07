"""Tests for the main application module."""

from importlib import import_module

from typer.testing import CliRunner

runner = CliRunner()


def test_import() -> None:
    """Placeholder test that ensures package import works as expected."""
    import_module("immersa_tesseract_inference")


def test_cli_run() -> None:
    """Test the CLI run command."""
    main_module = import_module("immersa_tesseract_inference.main")
    result = runner.invoke(main_module.app)
    assert result.exit_code == 0, result.output
    assert "Running Immersa Tesseract Inference pipeline" in result.stdout


def test_cli_help() -> None:
    """Test the CLI help command."""
    main_module = import_module("immersa_tesseract_inference.main")
    result = runner.invoke(main_module.app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "Usage" in result.stdout
