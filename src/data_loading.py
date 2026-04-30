import kagglehub
import shutil
import os
import pandas as pd


def download_data():
    """
    Downloads the dataset from Kaggle and saves it to data/raw.
    Handles kagglehub cache failures.
    """

    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists("data/raw/fetal_health.csv"):
        print("Dataset already exists. Skipping download.")
        return

    print("Downloading dataset...")

    path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")
    files = os.listdir(path)

    if not files:
        print("Empty cache detected. Clearing cache and retrying...")

        cache_dir = os.path.expanduser("~/.cache/kagglehub")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")
        files = os.listdir(path)

        if not files:
            raise ValueError("Download failed even after clearing cache.")

    for file in files:
        shutil.move(os.path.join(path, file), "data/raw/")

    print("Download completed successfully.")


def load_data(path):
    """
    Loads the dataset from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: DataFrame containing the loaded data.
    """
    df = pd.read_csv(path)

    print(f"Dataset loaded from: {path}")
    print(f"Shape: {df.shape}")

    return df


def log_raw_data(df):
    """
    Logs the raw dataset to Weights & Biases as an artifact.
    """

    import wandb

    temp_path = "temp_raw.csv"
    df.to_csv(temp_path, index=False)

    artifact = wandb.Artifact(
        name="fetal_health_raw",
        type="dataset",
        description="Raw dataset"
    )

    artifact.add_file(temp_path)

    wandb.log_artifact(artifact)

    os.remove(temp_path)