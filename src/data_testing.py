import pandas as pd

def test_no_missing_values(df: pd.DataFrame) -> None:
    total_missing = df.isnull().sum().sum()
    assert total_missing == 0, \
        f"TEST FAILED: {total_missing} missing values remain."
    print("PASS: no missing values.")


def test_target_values(df: pd.DataFrame, target_col: str, allowed_values) -> None:
    invalid_mask = ~df[target_col].isin(allowed_values)
    invalid_count = invalid_mask.sum()

    assert invalid_count == 0, \
        f"TEST FAILED: {invalid_count} unexpected values in '{target_col}': " \
        f"{df.loc[invalid_mask, target_col].unique()}"

    print(f"PASS: '{target_col}' contains only valid values.")


def test_no_duplicates(df: pd.DataFrame) -> None:
    assert df.duplicated().sum() == 0, "TEST FAILED: duplicatas ainda existem!"
    print("PASS: no duplicates.")