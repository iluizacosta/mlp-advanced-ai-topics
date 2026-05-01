import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset, DataLoader
from feature_selection import train_df_selected, test_df_selected

def prepare_dataloaders(train_df, test_df, target_col, batch_size=32):
    """
    Prepares PyTorch DataLoaders with proper standardization.

    - Fits scaler ONLY on training data (avoids data leakage)
    - Converts data to float32 tensors
    - Returns train and test DataLoaders
    """

    # ======================
    # 1. Separate features and target
    # ======================
    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values.astype(np.int64)

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values.astype(np.int64)

    # ======================
    # 2. Standardization
    # ======================
    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # ======================
    # 3. Convert to tensors
    # ======================
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.long)

    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

    # ======================
    # 4. Create datasets
    # ======================
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    # ======================
    # 5. Create dataloaders
    # ======================
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_dim = X_train.shape[1]

    return train_loader, test_loader, scaler, input_dim

train_loader, test_loader, scaler, input_dim = prepare_dataloaders(
    train_df_selected,
    test_df_selected,
    target_col="fetal_health",
    batch_size=32
)

print(f"Input dimension: {input_dim}")
print(f"Train batches: {len(train_loader)}")
print(f"Test batches: {len(test_loader)}")

for X, y in train_loader:
    print(f"Batch shape (X): {X.shape}")
    print(f"Batch shape (y): {y.shape}")
    break