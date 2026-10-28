set -eo pipefail
cd -- "$(dirname -- "$0")"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"$PWD/feature_env/bin/python" -B -m unittest test_contract132 test_model_input132 test_source_binding132 test_postglobal132 test_unique_support132 test_measured_refine132 test_subset_held132 test_subset_rigid132 test_admission132
"$PWD/feature_env/bin/python" -B test_crossmodal132.py
