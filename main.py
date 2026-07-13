import argparse

from src.experiments import run_all_experiments


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="predict_students_dropout_and_academic_success")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument(
        "--norm",
        type=str,
        default="zscore",
        choices=["mean", "median", "zscore", "none"],
    )
    parser.add_argument("--jl_eps", type=float, default=0.5)
    parser.add_argument("--rp_frac", type=float, default=0.1)
    parser.add_argument("--hidden", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--lambda_ent", type=float, default=0.0)
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--output_root", type=str, default="output")
    parser.add_argument(
        "--experiments",
        type=str,
        default="all",
        help="Comma-separated list among original_space,pca_only,uniform_multiview,adaptive_multiview, or all.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_all_experiments(parse_args())
