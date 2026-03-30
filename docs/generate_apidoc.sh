#!/usr/bin/env bash
set -euo pipefail

# Generate Sphinx API docs (run from project root or docs folder)
# Usage: ./docs/generate_apidoc.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

echo "Removing old generated API files..."
rm -rf docs/api

echo "Running sphinx-apidoc for package 'ScintSuite'..."
# Generate rst files for the ScintSuite package into docs/api
python -m sphinx.apidoc -o docs/api ScintSuite -f

echo "Done. You can now build docs:"
echo "  python -m pip install -r docs/requirements.txt"
echo "  python -m sphinx -b html docs docs/_build/html"
