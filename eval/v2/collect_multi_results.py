import argparse
import json
from pathlib import Path
import pandas as pd

def get_results(checkpoint_collection_dir: Path) -> pd.DataFrame:
    parent_dir = checkpoint_collection_dir
    result_files: list[Path] = []
    for d in parent_dir.iterdir():
        if d.is_dir():
            for sub_d in d.iterdir():
                if sub_d.name == "results.json":
                    result_files.append(sub_d)

    results = []
    for result_path in result_files:
        results.append(
            json.loads(result_path.read_text())
        )

    df = pd.json_normalize(results)


    # isolate these columns
    columns_of_interest = [
        "bps",
        "event_ppl",
        # test set loss Lakh
        "loss",
        "n",
        "k",
        "subsample",
        "total_params",
        "model_property_n_layer",
        "model_property_transformer_matrices",
        "model_property_lm_head",
    ]
    df = df[columns_of_interest]

    df["t"] = df["n"] + df["k"]

    # ignore positional and value embeddings
    df["p"] = df["model_property_transformer_matrices"] + df["model_property_lm_head"]

    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Multi Training Result Collector"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        required=True,
        help="Path to all models' enclosing path. Assumes results have been run already.",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    """

        PYTHONPATH=. python eval/v2/collect_multi_results.py --output_dir output/slurm_logs/319595/

    """
    args = parse_args()
    parent_dir = Path(args.output_dir)
    assert parent_dir.exists()
    assert parent_dir.is_dir()

    save_to = parent_dir / "collected.csv"

    df = get_results(parent_dir)
    df.to_csv(save_to, index=False)
    print(f"Saved results to: {save_to}")
