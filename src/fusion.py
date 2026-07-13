import numpy as np
from scipy.stats import mode
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    pairwise_distances,
    r2_score,
)
from sklearn.random_projection import GaussianRandomProjection, johnson_lindenstrauss_min_dim


def distance_matrix(X_train_view: np.ndarray, X_test_view: np.ndarray) -> np.ndarray:
    return pairwise_distances(X_test_view, X_train_view, metric="euclidean")


def normalize_dist_matrix(D: np.ndarray, norm: str, eps: float = 1e-12) -> np.ndarray:
    if norm == "none":
        return D
    if norm == "mean":
        return D / (D.mean(axis=1, keepdims=True) + eps)
    if norm == "median":
        return D / (np.median(D, axis=1, keepdims=True) + eps)
    if norm == "zscore":
        mu = D.mean(axis=1, keepdims=True)
        sigma = D.std(axis=1, keepdims=True)
        return (D - mu) / (sigma + eps)
    raise ValueError(f"Unknown normalization mode: {norm}")


def knn_predict(D_agg: np.ndarray, y_train: np.ndarray, k: int, task: str) -> np.ndarray:
    nn_indices = np.argsort(D_agg, axis=1)[:, :k]
    nn_labels = y_train[nn_indices]
    if task == "regression":
        return nn_labels.mean(axis=1)
    return mode(nn_labels, axis=1, keepdims=False).mode.ravel()


def compute_metrics(preds_np: np.ndarray, truth_np: np.ndarray, task: str) -> dict:
    if task == "regression":
        return {
            "rmse": float(np.sqrt(mean_squared_error(truth_np, preds_np))),
            "mae": float(mean_absolute_error(truth_np, preds_np)),
            "r2": float(r2_score(truth_np, preds_np)),
        }

    preds_cls = preds_np.astype(int)
    truth_cls = truth_np.astype(int)
    report = classification_report(truth_cls, preds_cls, output_dict=True)
    return {
        "accuracy": float(accuracy_score(truth_cls, preds_cls)),
        "f1_macro": float(f1_score(truth_cls, preds_cls, average="macro")),
        "f1_weighted": float(f1_score(truth_cls, preds_cls, average="weighted")),
        "per_class_report": report,
    }


def primary_metric(metrics: dict, task: str) -> float:
    return metrics["rmse"] if task == "regression" else metrics["accuracy"]


def is_better(a: dict, b: dict, task: str) -> bool:
    if task == "regression":
        return primary_metric(a, task) < primary_metric(b, task)
    return primary_metric(a, task) > primary_metric(b, task)


def get_feature_type_stats(info: dict, total: int) -> dict:
    stats = {"total": int(total), "numerical": None, "categorical": None}
    for k in ["n_num_features", "num_features", "numerical_features", "n_numerical_features"]:
        if k in info:
            stats["numerical"] = int(info[k])
            break
    for k in ["n_cat_features", "cat_features", "categorical_features", "n_categorical_features"]:
        if k in info:
            stats["categorical"] = int(info[k])
            break
    return stats


def build_projections(X_train: np.ndarray, y_train: np.ndarray, task: str, args):
    n_train = X_train.shape[0]
    n_features = X_train.shape[1]

    random_dim = johnson_lindenstrauss_min_dim(n_train, eps=args.jl_eps)
    random_dim = min(random_dim, int(n_features * args.rp_frac))
    random_dim = max(1, random_dim)

    svd_dim = max(1, min(random_dim, n_features - 1))

    projections = [
        PCA(n_components=0.99),
        GaussianRandomProjection(n_components=random_dim),
        TruncatedSVD(n_components=svd_dim, random_state=42),
    ]

    if task == "classification":
        n_classes = len(np.unique(y_train))
        lda_components = min(n_classes - 1, n_features)
        projections.insert(1, LinearDiscriminantAnalysis(n_components=lda_components))

    latent_spaces = []
    for proj in projections:
        if isinstance(proj, LinearDiscriminantAnalysis):
            X_lat = proj.fit_transform(X_train, y_train)
        else:
            X_lat = proj.fit_transform(X_train)
        latent_spaces.append((proj, X_lat))

    view_names = ["original"] + [type(proj).__name__ for proj, _ in latent_spaces]
    return latent_spaces, view_names, random_dim, svd_dim


def build_distance_views(
    X_train: np.ndarray,
    X_query: np.ndarray,
    latent_spaces: list,
    norm_mode: str,
    eps: float = 1e-12,
) -> list:
    views = []
    views.append(normalize_dist_matrix(distance_matrix(X_train, X_query), norm=norm_mode, eps=eps))
    for proj, X_train_lat in latent_spaces:
        X_query_lat = np.asarray(proj.transform(X_query))
        if X_query_lat.ndim == 1:
            X_query_lat = X_query_lat.reshape(-1, 1)
        views.append(
            normalize_dist_matrix(
                distance_matrix(X_train_lat, X_query_lat),
                norm=norm_mode,
                eps=eps,
            )
        )
    return views
