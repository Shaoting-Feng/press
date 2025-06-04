#!/usr/bin/env bash
set -euo pipefail

# Path to your input CSV:
INPUT_CSV="hotpotqa.csv"

# Derive the prefix (everything before “.csv”):
INPUT_PREFIX="${INPUT_CSV%.csv}"

# (Optional) Create a directory named after INPUT_PREFIX if it doesn't exist:
mkdir -p "${INPUT_PREFIX}"

# Array of compression rates to test:
RATES=(0.271428571 0.514285714 0.628571429)

# Loop through each rate and invoke the Python script
for RATE in "${RATES[@]}"; do
    OUTPUT_CSV="${INPUT_PREFIX}/results_rate_${RATE}.csv"
    echo "Running with compression rate=${RATE} → writing to ${OUTPUT_CSV}"
    python ${INPUT_PREFIX}/${INPUT_PREFIX}.py \
        --input-csv "${INPUT_PREFIX}/${INPUT_CSV}" \
        --output-csv "${OUTPUT_CSV}" \
        --compression-rate "${RATE}"
done

echo "All runs complete."

