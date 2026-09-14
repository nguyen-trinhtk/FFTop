#!/usr/bin/env bash
# Build fftop_bench, measure the config union, plot each comparison.
#
#   ./bench/bench-sweep.sh
#   ./bench/bench-sweep.sh -o log/custom-run
#   ./bench/bench-sweep.sh --only radix2-vs-radix4
#   ./bench/bench-sweep.sh --dry-run
set -euo pipefail

root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$root"

build_dir="${BUILD_DIR:-build}"
bench="${build_dir}/bench/fftop_bench"
python=".venv/bin/python3"

if [[ ! -x "$bench" ]]; then
  cmake -B "$build_dir" -DCMAKE_BUILD_TYPE=Release
  cmake --build "$build_dir" --target fftop_bench
fi

if [[ ! -x "$python" ]]; then
  python3 -m venv .venv
fi
export MPLCONFIGDIR="${MPLCONFIGDIR:-$root/.mplconfig}"
mkdir -p "$MPLCONFIGDIR"

"$python" -c "import matplotlib, yaml" 2>/dev/null || \
  "$python" -m pip install -r bench/requirements.txt

"$python" bench/bench.py --check --config bench/config/default.yaml
exec "$python" bench/bench.py --bench "$bench" --config bench/config/default.yaml "$@"
