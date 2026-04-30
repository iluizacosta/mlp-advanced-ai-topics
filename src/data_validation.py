import pandas as pd


def validate_no_missing_values(df: pd.DataFrame) -> None:
    """
    Ensure that the dataset contains no missing values.
    """
    total_missing = df.isnull().sum().sum()
    assert total_missing == 0, \
        f"VALIDATION FAILED: {total_missing} missing values remain."
    print("PASS: no missing values.")


def validate_target_values(
    df: pd.DataFrame,
    target_col: str,
    allowed_values
) -> None:
    """
    Ensure that the target column contains only valid values.
    """
    invalid_mask = ~df[target_col].isin(allowed_values)
    invalid_count = invalid_mask.sum()

    assert invalid_count == 0, \
        f"VALIDATION FAILED: {invalid_count} unexpected values in '{target_col}': " \
        f"{df.loc[invalid_mask, target_col].unique()}"

    print(f"PASS: '{target_col}' contains only valid values.")


def validate_no_duplicates(df: pd.DataFrame) -> None:
    """
    Ensure that the dataset contains no duplicate rows.
    """
    total_duplicates = df.duplicated().sum()
    assert total_duplicates == 0, \
        f"VALIDATION FAILED: {total_duplicates} duplicate rows found."
    print("PASS: no duplicates.")