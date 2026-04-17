import kagglehub
import shutil
import os
import pandas as pd
import wandb

def download_data():
    """
    Faz o download do dataset do Kaggle e salva em data/raw.
    Trata falhas de cache do kagglehub.
    """

    os.makedirs("data/raw", exist_ok=True)

    if os.path.exists("data/raw/fetal_health.csv"):
        print("Dataset já existe. Pulando download.")
        return

    print("Baixando dataset...")

    path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")
    files = os.listdir(path)

    if not files:
        print("Cache vazio detectado. Limpando cache e tentando novamente...")

        cache_dir = os.path.expanduser("~/.cache/kagglehub")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

        path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")
        files = os.listdir(path)

        if not files:
            raise ValueError("Falha no download mesmo após limpar o cache.")

    for file in files:
        shutil.move(os.path.join(path, file), "data/raw/")

    print("Download concluído com sucesso.")

def log_raw_data():
    """
    Loga o dataset bruto no W&B como artifact.
    """

    run = wandb.init(project="mlp-fetal-health", job_type="data_ingestion")

    artifact = wandb.Artifact(
        name="fetal_health_raw",
        type="dataset"
    )

    artifact.add_file("data/raw/fetal_health.csv")

    run.log_artifact(artifact)
    run.finish()