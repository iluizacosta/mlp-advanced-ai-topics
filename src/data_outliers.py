# import pandas as pd


# def remove_outliers_iqr(df: pd.DataFrame, target_col: str, multiplier: float = 3.0) -> pd.DataFrame:
#     """
#     Remove outliers usando o método IQR (Intervalo Interquartil).

#     Apenas colunas numéricas (exceto o target) são consideradas.

#     Args:
#         df: DataFrame original
#         target_col: nome da coluna alvo (não será usada no cálculo)
#         multiplier: fator do IQR (default = 1.5)

#     Returns:
#         DataFrame sem outliers
#     """

#     # Selecionar apenas features numéricas (sem o target)
#     numeric_cols = df.drop(columns=[target_col]).select_dtypes(
#         include=["int64", "float64"]
#     ).columns

#     # Calcular Q1, Q3 e IQR
#     Q1 = df[numeric_cols].quantile(0.25)
#     Q3 = df[numeric_cols].quantile(0.75)
#     IQR = Q3 - Q1

#     # Criar máscara para filtrar linhas sem outliers
#     mask = ~(
#         (df[numeric_cols] < (Q1 - multiplier * IQR)) |
#         (df[numeric_cols] > (Q3 + multiplier * IQR))
#     ).any(axis=1)

#     before = df.shape[0]
#     df_clean = df[mask]
#     after = df_clean.shape[0]

#     print(f"Outliers removidos: {before - after}")

#     return df_clean