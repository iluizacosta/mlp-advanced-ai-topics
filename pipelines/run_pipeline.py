import yaml
import wandb

from src.utils import set_seed
from src.data_loading import download_data, log_raw_data, load_data

# Carregar config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Aplicar seed
set_seed(config["seed"])

run = wandb.init(
    project="mlp-fetal-health",
    name="data_ingestion"
)

# Download dos dados
download_data()

# Carregar os dados
df = load_data(config["data"]["raw_path"])

# Log do dataset como artefato no W&B
log_raw_data(df)

