#!/usr/bin/env python3
"""Unified entry point for the eval awareness testbed.

Usage:
    python run.py list              # List available methods
    python run.py judge ...         # Judge transcripts
    python run.py analyze ...       # Analyze explanations
    python run.py experiment ...    # Run full experiment from config

Or install and use the CLI:
    pip install -e .
    eat list
    eat judge ...
"""

from eval_awareness_testbed.cli import app

if __name__ == "__main__":
    app()
