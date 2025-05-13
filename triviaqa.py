from transformers import pipeline
from kvpress import StreamingLLMPress
import time
import pandas as pd
import csv
import argparse

# Align model naming for both pipeline and (optionally) OpenAI engine
MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"
DEVICE = "cuda:0"

# Initialize the kv-press text-generation pipeline
pipe = pipeline(
    "kv-press-text-generation",
    model=MODEL_NAME,
    device=DEVICE,
    model_kwargs={"attn_implementation": "flash_attention_2"},
    torch_dtype="bfloat16"
)

def execute_kvpress_request_with_output(row, press) -> tuple[float, float, float, float, str]:
    """
    Execute a single summarisation request using the kv-press pipeline.

    Returns:
        start_time: float   – time when the request started (seconds)
        ttft: float         – time-to-first-token (seconds)
        finish_time: float  – total time taken (seconds)
        throughput: float   – characters per second (approx.)
        generated_text: str – the summary output
    """
    context = (
        f"This is user {row.index_in_dataset} in {row.dataset}.\n\n"
        "You are a precise, factual question‑answering assistant.  "
        "Below are example contexts (with their reference answers) to illustrate the desired format:\n\n"
        f"{row.context}\n\n"
        "Now, using that style, please supply only the correct Answer within 5 words.  "
        "**Do not include any heading or the word “Answer:” — just output the answer text.**"
    )
    question = row.input

    start_time = time.perf_counter()
    result = pipe(context, question=question, press=press)
    finish_time = time.perf_counter()

    generated_text = result.get("answer", "")
    ttft = finish_time - start_time
    duration = finish_time - start_time
    throughput = len(generated_text) / duration if duration > 0 else 0.0

    return start_time, ttft, duration, throughput, generated_text

def main(input_csv: str, output_csv: str, compression_rate: float):
    # Create StreamingLLMPress with the desired ratio
    press = StreamingLLMPress(compression_ratio=compression_rate)

    # Read input rows from CSV (must include columns: index_in_dataset, dataset, context, input)
    df = pd.read_csv(input_csv)
    df['index_in_dataset'] = list(range(len(df)))

    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index_in_dataset",
            "dataset",
            "start_time",
            "ttft",
            "finish_time",
            "throughput",
            "generated_text"
        ])
        for _, row in df.iterrows():
            start_time, ttft, finish_time, throughput, text = execute_kvpress_request_with_output(row, press)
            writer.writerow([
                row.index_in_dataset,
                row.dataset,
                start_time,
                ttft,
                finish_time,
                throughput,
                text.replace("\n", " ")
            ])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run kv-press summarisation on each row of an input CSV and save results to another CSV"
    )
    parser.add_argument(
        "--input-csv", type=str, default="samsum.csv",
        help="Path to the input CSV file"
    )
    parser.add_argument(
        "--output-csv", type=str, default="results.csv",
        help="Path to the output CSV file"
    )
    parser.add_argument(
        "--compression-rate", type=float, default=0.5,
        help="Compression ratio to use for StreamingLLMPress"
    )
    args = parser.parse_args()
    main(args.input_csv, args.output_csv, args.compression_rate)
