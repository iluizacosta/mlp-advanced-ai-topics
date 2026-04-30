import yaml
import wandb
import pandas as pd
from src.utils import set_seed
from src.data_loading import download_data, load_data, log_raw_data
from src.data_cleaning import (
    remove_duplicates,
    check_missing_values,
    save_clean_data,
    log_clean_data
)
from src.data_validation import (
    validate_no_missing_values,
    validate_target_values,
    validate_no_duplicates
)
from src.split_data import stratified_split, ks_test
from src.feature_selection import combined_feature_ranking


# ======================
# LOAD CONFIGURATION
# ======================
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

# Set random seed for reproducibility
set_seed(config["seed"])


# ======================
# DATA INGESTION
# ======================
# Download dataset if not already available
download_data()

# Load raw dataset
df = load_data(config["data"]["raw_path"])


# ======================
# LOG RAW DATA (W&B)
# ======================
# Log the raw dataset as an artifact for versioning

# Initialize W&B run
run = wandb.init(
    project="mlp-fetal-health",
    name="data_raw",
    config=config
)

log_raw_data(df)

# Finish W&B run
wandb.finish()


# ======================
# DATA CLEANING
# ======================
# Remove duplicate rows
df = remove_duplicates(df)

# Check for missing values
check_missing_values(df)


# Outlier removal was considered but not applied, as it removed a large portion of the dataset (approx. 50%!)


# ======================
# DATA VALIDATION (TESTING)
# ======================
# Ensure no missing values remain
validate_no_missing_values(df)

# Validate target values
validate_target_values(
    df,
    config["data"]["target_col"],
    [1, 2, 3]  # fetal_health classes
)

# Ensure no duplicate rows remain
validate_no_duplicates(df)


# ======================
# SAVE CLEAN DATA
# ======================
clean_path = save_clean_data(df)

# ======================
# LOG CLEAN DATA
# ======================

# Initialize W&B run
run = wandb.init(
    project="mlp-fetal-health",
    name="data_clean",
    config=config
)

log_clean_data(clean_path)

# Finish W&B run
wandb.finish()


# ======================
# TRAIN/TEST SPLIT
# ======================
# Perform stratified split to preserve class distribution
train_df, test_df = stratified_split(
    df,
    config["data"]["target_col"]
)


# ======================
# KS TEST (DISTRIBUTION VALIDATION)
# ======================
# Compare feature distributions between train and test sets
feature_cols = [
    c for c in train_df.columns
    if c != config["data"]["target_col"]
]

ks_results = ks_test(train_df, test_df, feature_cols)

# Convert results to DataFrame
comp_df = pd.DataFrame(ks_results).T.reset_index()
comp_df.columns = ["feature", "test", "statistic", "p_value"]

print("\nKS Test Results:")
print(comp_df)

# Initialize W&B run
run = wandb.init(
    project="mlp-fetal-health",
    name="data_split",
    config=config
)

# Log KS test results as table
comp_table = wandb.Table(dataframe=comp_df)
wandb.log({"distribution_comparison": comp_table})

# Log dataset sizes
wandb.summary.update({
    "train_size": len(train_df),
    "test_size": len(test_df)
})

# Log average p-value
wandb.summary["avg_p_value"] = comp_df["p_value"].mean()

# Finish run
wandb.finish()


# ======================
# SAVE TRAIN/TEST DATASETS
# ======================
train_path = "data/processed/train.csv"
test_path = "data/processed/test.csv"

train_df.to_csv(train_path, index=False)
test_df.to_csv(test_path, index=False)


# ======================
# LOG SPLIT ARTIFACT TO W&B
# ======================

# Initialize W&B run
run = wandb.init(
    project="mlp-fetal-health",
    name="data_split_FINAL_TEST",
    config=config
)

artifact = wandb.Artifact(
    name="fetal_health_split",
    type="dataset",
    description="Train/test split"
)

artifact.add_file(train_path)
artifact.add_file(test_path)

wandb.log_artifact(artifact)

# Log dataset sizes
wandb.summary.update({
    "train_size": len(train_df),
    "test_size": len(test_df)
})

# Finish W&B run
wandb.finish()


# ======================
# FEATURE SELECTION - RANKING
# ======================

ranking = combined_feature_ranking(train_df, config["data"]["target_col"])

print("\nFeature Ranking:")
print(ranking)

# Save ranking
ranking.to_csv("data/processed/feature_ranking.csv")
