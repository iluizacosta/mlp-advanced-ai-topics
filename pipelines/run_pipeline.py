import yaml

from src.utils import set_seed

# Carregar o config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Aplicar a seed
set_seed(config["seed"])