import pandas as pd
from src.data_cleaning import remove_duplicates, check_missing_values


def test_remove_duplicates():
    """
    Test if remove_duplicates correctly removes duplicate rows.
    """
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [3, 3, 4]
    })

    df_clean = remove_duplicates(df)

    # Should remove 1 duplicate
    assert len(df_clean) == 2
    assert df_clean.duplicated().sum() == 0


def test_remove_duplicates_no_change():
    """
    Test if the function does not modify the DataFrame when there are no duplicates.
    """
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    df_clean = remove_duplicates(df)

    assert len(df_clean) == 3


def test_check_missing_values():
    """
    Test if the function correctly detects missing values.
    """
    df = pd.DataFrame({
        "A": [1, None, 2],
        "B": [3, 4, None]
    })

    missing = check_missing_values(df)

    # Total missing values = 2
    total_missing = missing.sum()

    assert total_missing == 2