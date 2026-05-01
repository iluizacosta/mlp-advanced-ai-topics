import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor


def correlation_selection(df, target_col):
    """
    Computes absolute Spearman correlation between each feature and the target.
    """
    corr = df.corr(method="spearman", numeric_only=True)[target_col].drop(target_col)
    return corr.abs().sort_values(ascending=False)


def mutual_info_selection(df, target_col):
    """
    Estimates feature importance using mutual information.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    mi = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


def rf_feature_importance(df, target_col):
    """
    Computes feature importance using a Random Forest classifier.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]

    rf = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    )

    rf.fit(X, y)

    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(
        ascending=False
    )


def calculate_vif(df):
    """
    Computes Variance Inflation Factor (VIF) for each feature.
    """
    X = df.copy()

    vif_data = pd.DataFrame()
    vif_data["feature"] = X.columns
    vif_data["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]

    return vif_data.sort_values("VIF", ascending=False)


def remove_high_vif_features(df, threshold=10.0):
    """
    Iteratively removes features with VIF above the threshold.
    """
    X = df.copy()
    removed_features = []

    while True:
        vif_df = calculate_vif(X)
        max_vif = vif_df["VIF"].max()

        if max_vif > threshold:
            feature_to_drop = vif_df.iloc[0]["feature"]
            removed_features.append(feature_to_drop)
            X = X.drop(columns=[feature_to_drop])
        else:
            break

    final_vif = calculate_vif(X)

    return X.columns.tolist(), removed_features, final_vif


def combined_feature_selection(df, target_col, top_k=10, vif_threshold=10.0):
    """
    Full feature selection pipeline:
    - Ranking using correlation, mutual information, and Random Forest
    - Normalizes scores to [0, 1]
    - Selects top_k features
    - Removes highly collinear features using VIF
    """

    corr = correlation_selection(df, target_col)
    mi = mutual_info_selection(df, target_col)
    rf = rf_feature_importance(df, target_col)

    ranking = pd.DataFrame({
        "correlation": corr,
        "mutual_info": mi,
        "rf_importance": rf
    })

    ranking = ranking.fillna(0)

    ranking_norm = ranking.apply(
        lambda col: (col - col.min()) / (col.max() - col.min())
        if col.max() != col.min()
        else 0
    )

    ranking_norm["final_score"] = ranking_norm.mean(axis=1)
    ranking_norm = ranking_norm.sort_values("final_score", ascending=False)

    selected_features = ranking_norm.head(top_k).index.tolist()

    X_selected = df[selected_features]

    final_features, removed_features, final_vif = remove_high_vif_features(
        X_selected,
        threshold=vif_threshold
    )

    return ranking_norm, selected_features, final_features, removed_features, final_vif