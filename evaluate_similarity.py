import pandas as pd
from our_metrics import evaluate_answer, f1_score, codebleu_score
import os
import argparse

INPUT02 = '/home/ubuntu/st-prodstack-v/press/lcc_e/results_rate_0.628571429.csv'
INPUT03 = '/home/ubuntu/st-prodstack-v/press/lcc_e/results_rate_0.514285714.csv'
INPUT06 = '/home/ubuntu/st-prodstack-v/press/lcc_e/results_rate_0.271428571.csv'
INPUT0  = '/home/ubuntu/st-prodstack-v/LMCache/serve/results/Jun_19_1_coding/prefill/0.csv'

def main():
    parser = argparse.ArgumentParser(
        description="Compute ROUGE-L, F1, or CodeBLEU scores against a specified dataset in the reference CSV"
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help="Name of the dataset column value in the reference CSV (e.g. 'qmsum')"
    )
    parser.add_argument(
        '--metric',
        type=str,
        choices=['rouge', 'f1', 'codebleu'],
        default='rouge',
        help="Which metric to use: 'rouge' for ROUGE-L, 'f1' for F1 score, 'codebleu' for CodeBLEU"
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
        df = pd.read_csv(path).reset_index(drop=True)

        # Compute scores
        if args.metric == 'codebleu':
            scores = []
            for idx, row in df.iterrows():
                # 在 df0 中找到对应的 reference 行
                subset = df0[
                    (df0['dataset'] == target_dataset) &
                    (df0['index_in_dataset'] == row['index_in_dataset'])
                ]
                if subset.empty:
                    scores.append(-1)
                else:
                    ref_row = subset.iloc[0]

                    # 参考答案
                    ref = ref_row['answer']

                    # 语言标签从 df0 而不是 df
                    lang = ref_row.get('language', None)
                    if lang == 'csharp':
                        lang = 'c_sharp'

                    scores.append(codebleu_score(row['generated_text'], ref, lang))

            df['ROUGEL'] = scores

        else:
            # choose metric function
            metric_func = f1_score if args.metric == 'f1' else evaluate_answer
            df['ROUGEL'] = df.apply(
                lambda row: metric_func(
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
