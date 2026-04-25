import pandas as pd
from src.data_cleaning import remove_duplicates, check_missing_values


def test_remove_duplicates():
    """
    Testa se a função remove_duplicates remove corretamente linhas duplicadas.
    """
    df = pd.DataFrame({
        "A": [1, 1, 2],
        "B": [3, 3, 4]
    })

    df_clean = remove_duplicates(df)

    # Deve remover 1 duplicata
    assert len(df_clean) == 2
    assert df_clean.duplicated().sum() == 0


def test_remove_duplicates_no_change():
    """
    Testa se a função não altera o DataFrame quando não há duplicatas.
    """
    df = pd.DataFrame({
        "A": [1, 2, 3],
        "B": [4, 5, 6]
    })

    df_clean = remove_duplicates(df)

    assert len(df_clean) == 3


def test_check_missing_values():
    """
    Testa se a função detecta corretamente valores nulos.
    """
    df = pd.DataFrame({
        "A": [1, None, 2],
        "B": [3, 4, None]
    })

    total_missing = check_missing_values(df)

    # Existem 2 valores nulos
    assert total_missing == 2