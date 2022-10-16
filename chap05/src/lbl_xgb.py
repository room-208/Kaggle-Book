import pandas as pd
import xgboost as xgb
from sklearn import metrics, preprocessing


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    if 0:
        print(df)

    features = [f for f in df.columns if f not in ["if", "target", "kfold"]]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    for col in features:
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

    model = xgb.XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200)
    model.fit(X_train, y_train)

    valid_preds = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(auc)


if __name__ == "__main__":
    for fold in range(5):
        run(fold=fold)
