import joblib
import pandas as pd
from sklearn import metrics, tree
import os
import argparse
import model_dispatcher

import config


def run(fold, model_name):
    df = pd.read_csv(config.TRAINING_FILE)

    df_train = df[df["kfold"] != fold].reset_index(drop=True)
    df_valid = df[df["kfold"] == fold].reset_index(drop=True)

    if 0:
        print(df_train)
        print(df_valid)

    X_train = df_train.drop(columns=["label", "kfold"]).values
    y_train = df_train["label"].values
    X_valid = df_valid.drop(columns=["label", "kfold"]).values
    y_valid = df_valid["label"].values

    if 0:
        print(X_train, y_train)
        print(X_valid, y_valid)

    model = model_dispatcher.models[model_name]
    model.fit(X_train, y_train)
    preds = model.predict(X_valid)
    acc = metrics.accuracy_score(y_valid, preds)

    if 1:
        print(f"fold = {fold} acc = {acc}")

    joblib.dump(model, os.path.join(config.MODEL_OUTPUT, f"{model_name}_{fold}.bin"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--fold", type=int)
    parser.add_argument("--model_name", type=str)
    args = parser.parse_args()

    run(fold=args.fold, model_name=args.model_name)
