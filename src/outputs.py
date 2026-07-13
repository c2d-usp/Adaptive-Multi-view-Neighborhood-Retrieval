import csv
import json
import os

import numpy as np


CSV_COLUMNS = [
    "dataset",
    "accuracy_original_space",
    "accuracy_pca_only",
    "accuracy_uniform_multiview",
    "accuracy_adaptive_multiview",
]


def save_experiment(out_dir: str, experiment_name: str, payload: dict):
    exp_dir = os.path.join(out_dir, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)
    path = os.path.join(exp_dir, "metrics.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return path


def save_summary(out_dir: str, experiments: dict, task: str, dataset: str):
    summary = {"dataset": dataset, "task": task, "experiments": {}}

    for exp_name, results_list in experiments.items():
        if task == "classification":
            keys = ["accuracy", "f1_macro", "f1_weighted"]
        else:
            keys = ["rmse", "mae", "r2"]

        entry = {}
        for key in keys:
            vals = [r[key] for r in results_list if key in r]
            if not vals:
                continue
            entry[key] = {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals)) if len(vals) > 1 else None,
                "runs": vals,
            }

        time_vals = [r["time_sec"] for r in results_list if "time_sec" in r]
        if time_vals:
            entry["time_sec"] = {
                "mean": float(np.mean(time_vals)),
                "std": float(np.std(time_vals)) if len(time_vals) > 1 else None,
            }

        summary["experiments"][exp_name] = entry

    path = os.path.join(out_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return path


def save_results_csv(csv_path: str, dataset: str, metrics_by_exp: dict):
    file_exists = os.path.isfile(csv_path)

    def _acc(exp_key: str) -> str:
        m = metrics_by_exp.get(exp_key, {})
        v = m.get("accuracy")
        return f"{v:.6f}" if v is not None else ""

    row = {
        "dataset": dataset,
        "accuracy_original_space": _acc("original_space"),
        "accuracy_pca_only": _acc("pca_only"),
        "accuracy_uniform_multiview": _acc("uniform_multiview"),
        "accuracy_adaptive_multiview": _acc("adaptive_multiview"),
    }

    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)


def save_dataset_accuracy_comparison(out_dir: str, dataset: str, task: str, payloads_by_exp: dict):
    metric = "rmse" if task == "regression" else "accuracy"
    higher_is_better = task != "regression"
    experiments = {}

    for exp_name, payload in payloads_by_exp.items():
        if metric not in payload:
            continue
        entry = {
            metric: float(payload[metric]),
            "time_sec": float(payload["time_sec"]),
        }
        if "train_time_sec" in payload:
            entry["train_time_sec"] = float(payload["train_time_sec"])
        experiments[exp_name] = entry

    ranked = sorted(
        experiments.items(),
        key=lambda item: item[1][metric],
        reverse=higher_is_better,
    )
    ranking = [
        {"experiment": exp_name, metric: values[metric], "rank": rank}
        for rank, (exp_name, values) in enumerate(ranked, start=1)
    ]
    best_experiment = ranking[0]["experiment"] if ranking else None

    deltas = {}
    original = experiments.get("original_space", {}).get(metric)
    if original is not None:
        for exp_name, values in experiments.items():
            if exp_name == "original_space":
                continue
            absolute = values[metric] - original
            relative_percent = (absolute / original * 100.0) if original != 0 else None
            deltas[exp_name] = {
                "absolute": float(absolute),
                "relative_percent": float(relative_percent) if relative_percent is not None else None,
            }

    comparison = {
        "dataset": dataset,
        "task": task,
        "metric": metric,
        "higher_is_better": higher_is_better,
        "experiments": experiments,
        "ranking": ranking,
        "best_experiment": best_experiment,
        "deltas_vs_original_space": deltas,
    }

    path = os.path.join(out_dir, "accuracy_comparison.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    return path, comparison


def save_global_accuracy_comparison(output_root: str, dataset: str, comparison: dict):
    path = os.path.join(output_root, "accuracy_comparison.json")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            global_comparison = json.load(f)
    else:
        global_comparison = {"metric": comparison["metric"], "datasets": {}}

    global_comparison["metric"] = comparison["metric"]
    global_comparison.setdefault("datasets", {})

    metric = comparison["metric"]
    dataset_entry = {
        exp_name: values[metric]
        for exp_name, values in comparison["experiments"].items()
        if metric in values
    }
    dataset_entry["best_experiment"] = comparison["best_experiment"]
    global_comparison["datasets"][dataset] = dataset_entry

    with open(path, "w", encoding="utf-8") as f:
        json.dump(global_comparison, f, indent=2, ensure_ascii=False)
    return path
