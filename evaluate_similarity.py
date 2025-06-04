# This script evaluates the similarity of answers in a CSV file using the ROUGE metric.
import pandas as pd
from our_metrics import evaluate_answer, f1_score
import os
import argparse

INPUT02 = '/home/ubuntu/st-prodstack-v/press/qmsum/results_rate_0.628571429.csv'
INPUT03 = '/home/ubuntu/st-prodstack-v/press/qmsum/results_rate_0.514285714.csv'
INPUT06 = '/home/ubuntu/st-prodstack-v/press/qmsum/results_rate_0.271428571.csv'
INPUT0  = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/May_23_1_sum/prefill/0.csv'

def main():
    parser = argparse.ArgumentParser(
        description="Compute ROUGE‑L scores against a specified dataset in the reference CSV"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Name of the dataset column value in the reference CSV (e.g. 'qmsum')"
    )
    args = parser.parse_args()
    target_dataset = args.dataset

    # Load reference DataFrame (do not reset index or filter here)
    df0 = pd.read_csv(INPUT0)

    # Define which inputs to process and their short prefixes
    input_paths = [INPUT02, INPUT03, INPUT06]
    prefixes   = ['02',     '03',     '06']

    for path, prefix in zip(input_paths, prefixes):
        # Load & reset index on this CSV
        df = pd.read_csv(path)
        df = df.reset_index(drop=True)

        # Compute ROUGE‑L by looking up the reference answer in df0
        df['ROUGEL'] = df.apply(
            lambda row: evaluate_answer(
                row['generated_text'],
                df0.loc[
                    (df0['dataset'] == target_dataset) &
                    (df0['index_in_dataset'] == row['index_in_dataset']),
                    'answer'
                ].values[0]
            ),
            axis=1
        )

        # Build output filename by removing the numeric suffix from the original name
        original_base = os.path.splitext(os.path.basename(path))[0]
        base_prefix = original_base.rsplit('_', 1)[0]  # yields "results_rate"
        output_filename = f"{base_prefix}_{prefix}_processed.csv"
        output_path = os.path.join(target_dataset, output_filename)

        # Save
        df.to_csv(output_path, index=False)
        print(f"Processed CSV saved to {output_path}")

if __name__ == "__main__":
    main()
