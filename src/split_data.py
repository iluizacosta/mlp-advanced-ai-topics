import pandas as pd
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split


def stratified_split(df: pd.DataFrame, target_col: str, test_size=0.2, random_state=42):
    """
    Performs stratified split training and testing.
    """

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )

    train_df = pd.concat([X_train, y_train], axis=1)
    test_df = pd.concat([X_test, y_test], axis=1)

    return train_df, test_df