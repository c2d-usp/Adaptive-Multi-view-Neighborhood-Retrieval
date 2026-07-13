import json
import os
import time
import warnings

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder, StandardScaler

from src.adaptive import ViewWeighter, train_hypernetwork
from src.dataloader import get_dataset
from src.fusion import (
    build_distance_views,
    build_projections,
    compute_metrics,
    distance_matrix,
    get_feature_type_stats,
    is_better,
    knn_predict,
    normalize_dist_matrix,
)
from src.outputs import (
    save_dataset_accuracy_comparison,
    save_experiment,
    save_global_accuracy_comparison,
    save_results_csv,
    save_summary,
)


warnings.filterwarnings("ignore", category=RuntimeWarning)

PAPER_EXPERIMENTS = [
    "original_space",
    "pca_only",
    "uniform_multiview",
    "adaptive_multiview",
]


def _metrics_for_payload(metrics: dict) -> dict:
    return {k: v for k, v in metrics.items() if k != "per_class_report"}


def _log_metrics(log, experiment: str, elapsed: float, metrics: dict, label: str = "Time"):
    log(
        f"{experiment} | {label}: {elapsed:.4f}s | "
        f"Metrics: {json.dumps(_metrics_for_payload(metrics))}"
    )


def run_original_space(context):
    log = context["log"]
    log("\n-- Experiment: original_space ----------------------------------")
    t0 = time.perf_counter()

    D_orig = distance_matrix(context["X_train"], context["X_test"])
    orig_preds = knn_predict(D_orig, context["y_train"], context["k"], context["task"])
    orig_time = time.perf_counter() - t0

    orig_metrics = compute_metrics(
        context["to_orig"](orig_preds),
        context["truth_orig"],
        context["task"],
    )
    _log_metrics(log, "original_space", orig_time, orig_metrics)

    payload = {
        "experiment": "original_space",
        "dataset": context["dataset"],
        "task": context["task"],
        **_metrics_for_payload(orig_metrics),
        "time_sec": float(orig_time),
        "config": {"k": context["k"], "norm": context["norm_mode"]},
        "data": {
            "num_train_samples": context["n_train"],
            "num_test_samples": context["n_test"],
            "feature_types": get_feature_type_stats(context["info"], context["n_features"]),
        },
    }
    save_experiment(context["out_dir"], "original_space", payload)
    return payload, orig_metrics


def run_pca_only(context):
    log = context["log"]
    log("\n-- Experiment: pca_only ----------------------------------------")
    t0 = time.perf_counter()

    D_pca = normalize_dist_matrix(
        distance_matrix(context["X_train_pca"], context["X_test_pca"]),
        norm=context["norm_mode"],
        eps=context["eps"],
    )
    pca_preds = knn_predict(D_pca, context["y_train"], context["k"], context["task"])
    pca_time = time.perf_counter() - t0

    pca_metrics = compute_metrics(
        context["to_orig"](pca_preds),
        context["truth_orig"],
        context["task"],
    )
    _log_metrics(log, "pca_only", pca_time, pca_metrics)

    payload = {
        "experiment": "pca_only",
        "dataset": context["dataset"],
        "task": context["task"],
        **_metrics_for_payload(pca_metrics),
        "time_sec": float(pca_time),
        "pca_components": int(context["pca_proj"].n_components_),
        "config": {"k": context["k"], "norm": context["norm_mode"]},
        "data": {
            "num_train_samples": context["n_train"],
            "num_test_samples": context["n_test"],
            "feature_types": get_feature_type_stats(context["info"], context["n_features"]),
        },
    }
    save_experiment(context["out_dir"], "pca_only", payload)
    return payload, pca_metrics


def run_uniform_multiview(context):
    log = context["log"]
    log("\n-- Experiment: uniform_multiview -------------------------------")
    t0 = time.perf_counter()

    D_uniform = sum(context["D_views_test"]) / context["n_views"]
    uni_preds = knn_predict(D_uniform, context["y_train"], context["k"], context["task"])
    uni_time = time.perf_counter() - t0

    uni_metrics = compute_metrics(
        context["to_orig"](uni_preds),
        context["truth_orig"],
        context["task"],
    )
    _log_metrics(log, "uniform_multiview", uni_time, uni_metrics)

    payload = {
        "experiment": "uniform_multiview",
        "dataset": context["dataset"],
        "task": context["task"],
        **_metrics_for_payload(uni_metrics),
        "time_sec": float(uni_time),
        "n_views": context["n_views"],
        "view_names": context["view_names"],
        "config": {"k": context["k"], "norm": context["norm_mode"]},
        "data": {
            "num_train_samples": context["n_train"],
            "num_test_samples": context["n_test"],
            "feature_types": get_feature_type_stats(context["info"], context["n_features"]),
        },
    }
    save_experiment(context["out_dir"], "uniform_multiview", payload)
    return payload, uni_metrics


def run_adaptive_multiview(context):
    log = context["log"]
    args = context["args"]
    log("\n-- Experiment: adaptive_multiview (hypernetwork) ---------------")
    log(
        f"   hidden={args.hidden} | epochs={args.n_epochs} | batch={args.batch_size}"
        f" | lr={args.lr} | temperature={args.temperature} | lambda_ent={args.lambda_ent}"
    )

    t_train = time.perf_counter()
    weighter = ViewWeighter(context["n_features"], context["n_views"], hidden=args.hidden)

    loss_curve = train_hypernetwork(
        weighter=weighter,
        X_val=context["X_val"],
        D_views_val=context["D_views_val"],
        y_train=context["y_train"],
        y_val=context["y_val"],
        k=context["k"],
        task=context["task"],
        temperature=args.temperature,
        lambda_ent=args.lambda_ent,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        device=context["device"],
        eps=context["eps"],
    )
    train_time = time.perf_counter() - t_train
    log(f"Training time: {train_time:.4f}s")
    log(
        "Loss curve (every 10 epochs): "
        f"{[round(loss_curve[i], 5) for i in range(0, len(loss_curve), 10)]}"
    )

    t_inf = time.perf_counter()
    weighter.eval()
    with torch.no_grad():
        X_test_t = torch.tensor(context["X_test"], dtype=torch.float32, device=context["device"])
        alpha_test = weighter(X_test_t).cpu().numpy()

    D_stack_test = np.stack(context["D_views_test"], axis=0)
    D_dynamic = np.einsum("iv,vij->ij", alpha_test, D_stack_test)
    adp_preds = knn_predict(D_dynamic, context["y_train"], context["k"], context["task"])
    inf_time = time.perf_counter() - t_inf

    adp_metrics = compute_metrics(
        context["to_orig"](adp_preds),
        context["truth_orig"],
        context["task"],
    )
    _log_metrics(log, "adaptive_multiview", inf_time, adp_metrics, label="Inference time")

    alpha_mean = alpha_test.mean(axis=0)
    alpha_std = alpha_test.std(axis=0)
    log(
        "Mean alpha: "
        + str(
            {
                context["view_names"][v]: round(float(alpha_mean[v]), 4)
                for v in range(context["n_views"])
            }
        )
    )
    log(
        "Std alpha: "
        + str(
            {
                context["view_names"][v]: round(float(alpha_std[v]), 4)
                for v in range(context["n_views"])
            }
        )
    )

    payload = {
        "experiment": "adaptive_multiview",
        "dataset": context["dataset"],
        "task": context["task"],
        **_metrics_for_payload(adp_metrics),
        "time_sec": float(inf_time),
        "train_time_sec": float(train_time),
        "n_views": context["n_views"],
        "view_names": context["view_names"],
        "alpha_mean": {
            context["view_names"][v]: float(alpha_mean[v]) for v in range(context["n_views"])
        },
        "alpha_std": {
            context["view_names"][v]: float(alpha_std[v]) for v in range(context["n_views"])
        },
        "loss_curve": loss_curve,
        "config": {
            "k": context["k"],
            "norm": context["norm_mode"],
            "jl_eps": args.jl_eps,
            "rp_frac": args.rp_frac,
            "svd_dim": context["svd_dim"],
            "hidden": args.hidden,
            "n_epochs": args.n_epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "temperature": args.temperature,
            "lambda_ent": args.lambda_ent,
            "device": str(context["device"]),
        },
        "data": {
            "num_train_samples": context["n_train"],
            "num_val_samples": int(context["X_val"].shape[0]),
            "num_test_samples": context["n_test"],
            "feature_types": get_feature_type_stats(context["info"], context["n_features"]),
        },
    }
    save_experiment(context["out_dir"], "adaptive_multiview", payload)
    return payload, adp_metrics


def _selected_experiments(experiments_arg: str) -> list:
    if experiments_arg == "all":
        return PAPER_EXPERIMENTS.copy()

    selected = [e.strip() for e in experiments_arg.split(",") if e.strip()]
    unknown = [e for e in selected if e not in PAPER_EXPERIMENTS]
    if unknown:
        raise ValueError(f"Unknown experiment(s): {', '.join(unknown)}")
    return selected


def run_all_experiments(args):
    dataset = args.dataset
    k = args.k
    norm_mode = args.norm
    eps = 1e-12

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = os.path.join(args.output_root, dataset)
    os.makedirs(out_dir, exist_ok=True)

    log_file = open(os.path.join(out_dir, "log.txt"), "a", encoding="utf-8")

    def log(*msgs):
        msg = " ".join(str(m) for m in msgs)
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    try:
        log(f"Device: {device}")

        train_val_data, test_data, info = get_dataset(dataset, args.data_root)

        task_type = str(info.get("task_type", "")).lower() if isinstance(info, dict) else ""
        task = "classification" if task_type in {"binclass", "multiclass"} else "regression"
        log(f"Task type detected: {task}")

        X_train = train_val_data[0]["train"]
        X_val = train_val_data[0].get("val", None)
        y_train = train_val_data[2]["train"]
        y_val = train_val_data[2].get("val", None)
        X_test = test_data[0]["test"]
        y_test = test_data[2]["test"]

        x_scaler = StandardScaler()
        X_train = x_scaler.fit_transform(X_train)
        X_test = x_scaler.transform(X_test)
        if X_val is not None:
            X_val = x_scaler.transform(X_val)

        if task == "regression":
            y_scaler = StandardScaler()
            y_train = y_scaler.fit_transform(y_train.reshape(-1, 1)).ravel()
            y_test = y_scaler.transform(y_test.reshape(-1, 1)).ravel()
            if y_val is not None:
                y_val = y_scaler.transform(y_val.reshape(-1, 1)).ravel()
        else:
            y_scaler = LabelEncoder()
            y_train = y_scaler.fit_transform(y_train)
            y_test = y_scaler.transform(y_test)
            if y_val is not None:
                y_val = y_scaler.transform(y_val)

        log(dataset, X_train.shape, X_test.shape)

        n_train = X_train.shape[0]
        n_test = X_test.shape[0]
        n_features = X_train.shape[1]

        latent_spaces, view_names, random_dim, svd_dim = build_projections(
            X_train,
            y_train,
            task,
            args,
        )
        log(f"Random projection dim: {random_dim} | SVD dim: {svd_dim}")

        pca_proj = PCA(n_components=0.99)
        X_train_pca = pca_proj.fit_transform(X_train)
        X_test_pca = pca_proj.transform(X_test)

        D_views_test = build_distance_views(X_train, X_test, latent_spaces, norm_mode, eps)
        D_views_val = build_distance_views(X_train, X_val, latent_spaces, norm_mode, eps)
        n_views = len(D_views_test)
        log(f"Views: {n_views} | Names: {view_names}")

        def to_orig(preds):
            if task == "regression":
                return y_scaler.inverse_transform(preds.reshape(-1, 1)).ravel()
            return preds

        truth_orig = to_orig(y_test)
        context = {
            "args": args,
            "dataset": dataset,
            "k": k,
            "norm_mode": norm_mode,
            "eps": eps,
            "device": device,
            "out_dir": out_dir,
            "log": log,
            "info": info,
            "task": task,
            "X_train": X_train,
            "X_val": X_val,
            "X_test": X_test,
            "X_train_pca": X_train_pca,
            "X_test_pca": X_test_pca,
            "y_train": y_train,
            "y_val": y_val,
            "y_test": y_test,
            "to_orig": to_orig,
            "truth_orig": truth_orig,
            "n_train": n_train,
            "n_test": n_test,
            "n_features": n_features,
            "pca_proj": pca_proj,
            "D_views_test": D_views_test,
            "D_views_val": D_views_val,
            "n_views": n_views,
            "view_names": view_names,
            "svd_dim": svd_dim,
        }

        runners = {
            "original_space": run_original_space,
            "pca_only": run_pca_only,
            "uniform_multiview": run_uniform_multiview,
            "adaptive_multiview": run_adaptive_multiview,
        }

        all_experiments = {}
        metrics_by_exp = {}
        payloads_by_exp = {}
        for experiment in _selected_experiments(args.experiments):
            payload, metrics = runners[experiment](context)
            all_experiments[experiment] = [payload]
            metrics_by_exp[experiment] = metrics
            payloads_by_exp[experiment] = payload

        if "adaptive_multiview" in metrics_by_exp:
            log("\n-- Comparisons --------------------------------------------------")
            for baseline in ["original_space", "pca_only", "uniform_multiview"]:
                if baseline in metrics_by_exp:
                    result = (
                        "WINNER"
                        if is_better(metrics_by_exp["adaptive_multiview"], metrics_by_exp[baseline], task)
                        else "LOSER"
                    )
                    log(f"Adaptive vs {baseline}: {result}")

        summary_path = save_summary(out_dir, all_experiments, task, dataset)
        log(f"\nSummary saved to {summary_path}")

        csv_path = os.path.join(args.output_root, "results.csv")
        save_results_csv(csv_path, dataset, metrics_by_exp)
        log(f"CSV row appended -> {csv_path}")

        comparison_path, comparison = save_dataset_accuracy_comparison(
            out_dir,
            dataset,
            task,
            payloads_by_exp,
        )
        log(f"Dataset comparison saved to {comparison_path}")

        global_comparison_path = save_global_accuracy_comparison(
            args.output_root,
            dataset,
            comparison,
        )
        log(f"Global comparison saved to {global_comparison_path}")

        return {
            "summary_path": summary_path,
            "csv_path": csv_path,
            "comparison_path": comparison_path,
            "global_comparison_path": global_comparison_path,
        }
    finally:
        log_file.close()
