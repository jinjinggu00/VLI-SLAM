set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1
"$PWD/feature_env/bin/python" -B -m unittest test_source_binding132
