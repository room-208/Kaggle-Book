import pandas as pd
from sklearn import metrics, preprocessing
from sklearn.neural_network import MLPClassifier


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

    df = df.drop(columns=num_cols)

    features = [f for f in df.columns if f not in ["target", "kfold"]]

    for col in features:
        df[col] = df[col].astype(str).fillna("NONE")

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    if 0:
        print(df_train)
        print(df_valid)

    ohe = preprocessing.OneHotEncoder()
    ohe.fit(df[features].values)

    X_train = ohe.transform(df_train[features].values)
    y_train = df_train["target"].values
    X_valid = ohe.transform(df_valid[features].values)
    y_valid = df_valid["target"].values

    model = MLPClassifier(
        hidden_layer_sizes=(100, 100, 100, 100),
        activation="relu",
        solver="adam",
        batch_size="auto",
        learning_rate="constant",
        learning_rate_init=0.001,
        max_iter=100,
        shuffle=True,
        random_state=None,
        verbose=True,
        # warm_start=True,
        early_stopping=True,
    )
    model.fit(X_train, y_train)

    valid_preds = model.predict_proba(X_valid)[:, 1]

    auc = metrics.roc_auc_score(y_valid, valid_preds)

    print(auc)


if __name__ == "__main__":
    for fold in range(5):
        run(fold=fold)
