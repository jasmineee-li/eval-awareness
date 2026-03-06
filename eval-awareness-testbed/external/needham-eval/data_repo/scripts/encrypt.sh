#!/usr/bin/env bash
set -euo pipefail

# Change to project root directory
cd "$(dirname "$0")/.."

PW="isthisreallythepassword"
zip -reP "$PW" dataset.zip dataset.json mcq_transcripts/
rm dataset.json
echo "🔒  dataset.zip created"

