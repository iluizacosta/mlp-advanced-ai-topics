import yaml
import wandb

from src.utils import set_seed
from src.data_loading import download_data, load_data
from src.data_cleaning import (
    remove_duplicates,
    check_missing_values,
    save_clean_data,
    log_clean_data
)
from src.data_testing import (
    test_no_missing_values,
    test_target_values,
    test_no_duplicates
)
from src.split_data import stratified_split, ks_test
from src.train import split_features_target

# Carregar config
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Aplicar seed
set_seed(config["seed"])

run = wandb.init(
    project="mlp-fetal-health",
    name="data_cleaning"
)

# Download dos dados
download_data()

# Carregar os dados
df = load_data(config["data"]["raw_path"])

# Limpeza dos dados
df = remove_duplicates(df)
check_missing_values(df)

# outliers (não foi utilizado por remover extensas quantidades de dados)
# df = remove_outliers_iqr(
#    df,
#    config["data"]["target_col"]
# )

# Depois do cleaning
test_no_missing_values(df)
test_no_duplicates(df)

test_target_values(
    df,
    config["data"]["target_col"],
    [1, 2, 3]  # fetal_health
)

# Salvar clean
clean_path = save_clean_data(df)

# Logar clean
log_clean_data(clean_path)

wandb.finish()
