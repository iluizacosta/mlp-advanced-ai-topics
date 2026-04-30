import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier


# ======================
# 1. CORRELATION
# ======================
def correlation_selection(df, target_col):
    corr = df.corr()[target_col].drop(target_col)
    return corr.abs().sort_values(ascending=False)


# ======================
# 2. MUTUAL INFORMATION
# ======================
def mutual_info_selection(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    mi = mutual_info_classif(X, y, random_state=42)
    return pd.Series(mi, index=X.columns).sort_values(ascending=False)


# ======================
# 3. RANDOM FOREST
# ======================
def rf_feature_importance(df, target_col):
    X = df.drop(columns=[target_col])
    y = df[target_col]

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    return pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)


# ======================
# 4. COMBINED RANKING
# ======================
def combined_feature_ranking(df, target_col):
    corr = correlation_selection(df, target_col)
    mi = mutual_info_selection(df, target_col)
    rf = rf_feature_importance(df, target_col)

    ranking = pd.DataFrame({
        "correlation": corr,
        "mutual_info": mi,
        "rf_importance": rf
    })

    # Normalize scores
    ranking = (ranking - ranking.min()) / (ranking.max() - ranking.min())

    ranking["final_score"] = ranking.mean(axis=1)

    return ranking.sort_values("final_score", ascending=False)