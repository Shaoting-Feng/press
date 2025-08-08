#!/usr/bin/env bash
set -euo pipefail

# Path to your input CSV:
INPUT_CSV="qmsum.csv"

# Array of compression rates to test:
RATES=(0.271428571 0.514285714 0.628571429)

# Loop through each rate and invoke the Python script
for RATE in "${RATES[@]}"; do
    OUTPUT_CSV="results_rate_${RATE}.csv"
    echo "Running with compression rate=${RATE} → writing to ${OUTPUT_CSV}"
    python qmsum.py \
        --input-csv "${INPUT_CSV}" \
        --output-csv "${OUTPUT_CSV}" \
        --compression-rate "${RATE}"
done

echo "All runs complete."
