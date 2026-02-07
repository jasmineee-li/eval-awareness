#!/usr/bin/env bash
set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/.."

PW="isthisreallythepassword"
unzip -P "$PW" dataset.zip
echo "✅  dataset extracted."
