import kagglehub
import shutil
import os

def download_data():
    """
    Download do dataset do Kaggle e salva em data/raw.
    Evita download duplicado.
    """

    if os.path.exists("data/raw/fetal_health.csv"):
        print("Dataset já existe. Pulando download.")
        return

    path = kagglehub.dataset_download("andrewmvd/fetal-health-classification")

    os.makedirs("data/raw", exist_ok=True)

    for file in os.listdir(path):
        shutil.move(os.path.join(path, file), "data/raw/")