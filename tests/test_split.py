import pandas as pd

def ks_test(train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list) -> dict:
    """
    Compares distributions using KS Test only (Kolmogorov-Smirnov).
    """

    results = {}

    for col in columns:
        train_vals = train_df[col].dropna()
        test_vals = test_df[col].dropna()

        stat, p = ks_2samp(train_vals, test_vals)

        results[col] = {
            "test": "KS",
            "statistic": stat,
            "p_value": p
        }

    return results
