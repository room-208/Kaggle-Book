import pandas as pd
from scipy import sparse
from sklearn import decomposition, ensemble, metrics, preprocessing


def run(fold):
    df = pd.read_csv("../input/train_folds.csv")

    if 0:
        print(df)

    features = [f for f in df.columns if f not in ["if", "target", "kfold"]]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    if 0:
        print(df_train)
        print(df_valid)

    ohe = preprocessing.OneHotEncoder()

    full_data = pd.concat([df_train[features], df_valid[features]])

    ohe.fit(full_data[features].values)

    X_train = ohe.transform(df_train[features].values)
    y_train = df_train["target"].values
    X_valid = ohe.transform(df_valid[features].values)
    y_valid = df_valid["target"].values

    svd = decomposition.TruncatedSVD(n_components=120)

    full_sparse = sparse.vstack((X_train, X_valid))
    svd.fit(full_sparse)

    X_train = svd.transform(X_train)
    X_valid = svd.transform(X_valid)

    model = ensemble.RandomForestClassifier(n_jobs=-1)
    model.fit(X_train, y_train)

    valid_preds = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(auc)


if __name__ == "__main__":
    for fold in range(5):
        run(fold=fold)
