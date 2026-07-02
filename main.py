import argparse
from pathlib import Path

import pandas as pd

from feature_extraction import extract_features_for_all
from lisa.lisa_models import (
    train_and_evaluate_lisa_models,
    train_and_evaluate_lisa_models_ratios,
)
from sota.bert import evaluate_bert, evaluate_bert_on_ratios
from sota.detectgpt import evaluate_detectgpt, evaluate_detectgpt_on_ratios
from sota.gptzero import evaluate_gptzero, evaluate_gptzero_on_ratios
from sota.roberta import evaluate_roberta, evaluate_roberta_on_ratios


DEFAULT_SUBSET_SIZES = [10_000, 20_000]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run LISA experiments and optional baseline comparisons."
    )
    parser.add_argument(
        "--dataset",
        default="data/xsum.csv",
        help="Path to the input CSV file.",
    )
    parser.add_argument(
        "--text-column",
        default=None,
        help="Name of the text column. If omitted, the script tries 'text' and 'texts'.",
    )
    parser.add_argument(
        "--label-column",
        default=None,
        help="Name of the label column. If omitted, the script tries 'label' and 'labels'.",
    )
    parser.add_argument(
        "--subset-sizes",
        nargs="*",
        type=int,
        default=DEFAULT_SUBSET_SIZES,
        help="Additional leading subset sizes to evaluate after the full dataset.",
    )
    parser.add_argument(
        "--skip-sota",
        action="store_true",
        help="Skip BERT, RoBERTa, GPTZero-style, and DetectGPT-style baselines.",
    )
    parser.add_argument(
        "--skip-ratio-evals",
        action="store_true",
        help="Skip class-ratio experiments.",
    )
    return parser.parse_args()


def resolve_column_name(dataframe, explicit_name, candidates, column_role):
    if explicit_name is not None:
        if explicit_name not in dataframe.columns:
            raise ValueError(
                f"{column_role} column '{explicit_name}' was not found. "
                f"Available columns: {list(dataframe.columns)}"
            )
        return explicit_name

    for candidate in candidates:
        if candidate in dataframe.columns:
            return candidate

    raise ValueError(
        f"Could not infer the {column_role} column. "
        f"Tried {candidates}. Available columns: {list(dataframe.columns)}"
    )


def load_dataset(dataset_path, text_column=None, label_column=None):
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    print(f"Loading dataset from {dataset_path}")
    data = pd.read_csv(dataset_path)

    text_column = resolve_column_name(
        data, text_column, ["text", "texts"], "text"
    )
    label_column = resolve_column_name(
        data, label_column, ["label", "labels"], "label"
    )

    texts = data[text_column].fillna("").astype(str).tolist()
    labels = data[label_column].astype(int).tolist()

    print(f"Loaded {len(texts)} rows")
    print(f"Using text column '{text_column}' and label column '{label_column}'")

    return texts, labels


def iter_evaluation_sets(full_features, texts, labels, subset_sizes):
    yield "full dataset", full_features, texts, labels

    seen_sizes = set()
    for size in subset_sizes:
        if size <= 0:
            continue

        capped_size = min(size, len(labels))
        if capped_size >= len(labels) or capped_size in seen_sizes:
            continue

        seen_sizes.add(capped_size)
        yield (
            f"first {capped_size} samples",
            full_features[:capped_size],
            texts[:capped_size],
            labels[:capped_size],
        )


def evaluate_sota_full(texts, labels):
    print("\tBERT:")
    print(evaluate_bert(texts, labels))
    print("\tRoBERTa:")
    print(evaluate_roberta(texts, labels))
    print("\tGPTZero:")
    print(evaluate_gptzero(texts, labels))
    print("\tDetectGPT:")
    print(evaluate_detectgpt(texts, labels))


def evaluate_sota_ratios(texts, labels):
    print("\tBERT:")
    print(evaluate_bert_on_ratios(texts, labels))
    print("\tRoBERTa:")
    print(evaluate_roberta_on_ratios(texts, labels))
    print("\tGPTZero:")
    print(evaluate_gptzero_on_ratios(texts, labels))
    print("\tDetectGPT:")
    print(evaluate_detectgpt_on_ratios(texts, labels))


def evaluate_lisa_full(features, labels):
    lisa_results = train_and_evaluate_lisa_models(features, labels)
    for model_name, report in lisa_results.items():
        print(f"\nResults for {model_name}:")
        print(report)


def evaluate_lisa_ratios(features, labels):
    lisa_results = train_and_evaluate_lisa_models_ratios(features, labels)
    for model_name, report in lisa_results.items():
        print(f"\nResults for {model_name}:")
        print(report)


def main():
    args = parse_args()

    texts, labels = load_dataset(
        args.dataset,
        text_column=args.text_column,
        label_column=args.label_column,
    )

    print("Extracting features")
    full_features, _ = extract_features_for_all(texts)

    evaluation_sets = list(
        iter_evaluation_sets(full_features, texts, labels, args.subset_sizes)
    )

    for set_name, set_features, set_texts, set_labels in evaluation_sets:
        print(f"Evaluating without ratios ({set_name})")
        if not args.skip_sota:
            evaluate_sota_full(set_texts, set_labels)
        evaluate_lisa_full(set_features, set_labels)

    if args.skip_ratio_evals:
        return

    for set_name, set_features, set_texts, set_labels in evaluation_sets:
        print(f"Evaluating with ratios ({set_name})")
        if not args.skip_sota:
            evaluate_sota_ratios(set_texts, set_labels)
        evaluate_lisa_ratios(set_features, set_labels)


if __name__ == "__main__":
    main()
