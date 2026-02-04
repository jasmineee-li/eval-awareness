"""Tests for the CLI."""

import pytest
from typer.testing import CliRunner

from eval_awareness_testbed.cli import app

runner = CliRunner()


def test_cli_version():
    """Test that version command works."""
    result = runner.invoke(app, ["version"])
    assert result.exit_code == 0
    assert "eval-awareness-testbed" in result.stdout


def test_cli_list():
    """Test that list command shows available components."""
    result = runner.invoke(app, ["list"])
    assert result.exit_code == 0
    # Should show evals, judges, and analyzers tables
    assert "Evals" in result.stdout or "evals" in result.stdout.lower()


def test_cli_help():
    """Test that help is displayed."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.stdout.lower()
    assert "judge" in result.stdout.lower()
