#!/bin/bash
set -euo pipefail
exec bash "$(dirname "${BASH_SOURCE[0]}")/run_gaussian_diag.sh" --variant noadapter "$@"
