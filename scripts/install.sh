#!/usr/bin/env bash
# scripts/install.sh — thin wrapper around the root setup.sh
# Usage: bash scripts/install.sh [--cpu]
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$SCRIPT_DIR/../setup.sh" "$@"
