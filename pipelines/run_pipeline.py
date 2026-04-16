import yaml

from src.utils import set_seed
from src.data_loading import download_data

# Carregar config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Aplicar seed
set_seed(config["seed"])

# Download dos dados
download_data()