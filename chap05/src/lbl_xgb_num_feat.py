import itertools
import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing


def feature_engineering(df, cat_cols):
    combi = list(itertools.combinations(cat_cols, 2))
    for c1, c2 in combi:
        df.loc[:, c1 + "_" + c2] = df[c1].astype(str) + "_" + df[c2].astype(str)
    return df


def run(fold):
    df = pd.read_csv("../input/adult_folds.csv")

    if 0:
        print(df)

    num_cols = [
        "fnlwgt",
        "age",
        "capitalgain",
        "capitalloss",
        "hoursperweek",
    ]

    cat_cols = [
        c for c in df.columns if c not in num_cols and c not in ["target", "kfold"]
    ]

    df = feature_engineering(df, cat_cols)

    features = [f for f in df.columns if f not in ["target", "kfold"]]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    for col in features:
        if col not in num_cols:
            lbl = preprocessing.LabelEncoder()
            lbl.fit(df[col].values)
            df[col] = lbl.transform(df[col].values)

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    if 0:
        print(df_train)
        print(df_valid)

    X_train = df_train[features].values
    y_train = df_train["target"].values
    X_valid = df_valid[features].values
    y_valid = df_valid["target"].values

    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7)
    # model = xgb.XGBClassifier(n_jobs=-1)
    model.fit(X_train, y_train)

    valid_preds = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(auc)


if __name__ == "__main__":
    for fold in range(5):
        run(fold=fold)
