import pandas as pd
import catboost as cb
from catboost import CatBoostClassifier
from catboost import Pool
from sklearn import metrics, preprocessing


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

    train_pool = Pool(X_train, label=y_train)
    valid_pool = Pool(X_valid, label=y_valid)

    # model = xgb.XGBClassifier(n_jobs=-1, max_depth=7, n_estimators=200)
    model = CatBoostClassifier()
    model.fit(train_pool, eval_set=[valid_pool], use_best_model=True, verbose=0)

    valid_preds = model.predict_proba(valid_pool)[:, 1]
    print(valid_preds)

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(auc)


if __name__ == "__main__":
    for fold in range(5):
        run(fold=fold)
