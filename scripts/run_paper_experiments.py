import subprocess
import sys


PAPER_DATASETS = [
    "adult",
    "Bank_Customer_Churn_Dataset",
    "California-Housing-Classification",
    "electricity",
    "house_16H",
    "wine",
    "car-evaluation",
    "letter",
]


def main():
    for dataset in PAPER_DATASETS:
        subprocess.run([sys.executable, "main.py", "--dataset", dataset], check=True)


if __name__ == "__main__":
    main()
