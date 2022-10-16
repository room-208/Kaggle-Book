import pandas as pd
import seaborn as sns
from sklearn import model_selection
import matplotlib.pyplot as plt

if __name__ == "__main__":
    df = pd.read_csv("../input/train.csv")

    df["kfold"] = -1
    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)

    if 0:
        print(df)

    X = df.drop(columns=["target", "kfold"]).values
    y = df["kfold"].values

    for fold, (trn_, val_) in enumerate(kf.split(X=X, y=y)):
        df.loc[val_, "kfold"] = fold

    if 1:
        print(df)

    if 1:
        b = sns.countplot(x="target", data=df)
        plt.show()

        for fold in range(5):
            b = sns.countplot(x="target", data=df[df["kfold"] == fold])
            plt.show()

    df.to_csv("../input/train_folds.csv", index=False)
