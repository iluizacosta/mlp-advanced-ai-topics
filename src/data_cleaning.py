import wandb

def remove_duplicates(df):
    """
    Remove linhas duplicadas do dataset.
    """
    before = df.shape[0]

    df = df.drop_duplicates()

    after = df.shape[0]

    print(f"Duplicatas removidas: {before - after}")

    return df

def check_missing_values(df):
    """
    Verifica valores ausentes no dataset.
    """
    missing = df.isnull().sum()

    print("\nValores nulos por coluna:")
    print(missing)

    total_missing = missing.sum()

    print(f"\nTotal de valores nulos: {total_missing}")

    return total_missing

def save_clean_data(df):
    """
    Saves the clean dataset.
    """
    import os

    os.makedirs("data/processed", exist_ok=True)

    path = "data/processed/fetal_health_clean.csv"
    df.to_csv(path, index=False)

    print(f"Cleaned dataset saved at: {path}")

    return path

def log_clean_data(path):
    """
    Logs the clean data as an Artifact.
    """

    artifact = wandb.Artifact(
        name="fetal_health_clean",
        type="dataset",
        description="Cleaned dataset after preprocessing"
    )

    artifact.add_file(path)

    wandb.log_artifact(artifact)