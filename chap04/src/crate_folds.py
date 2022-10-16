import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import datasets, model_selection

if __name__ == '__main__':
    data = datasets.fetch_openml("mnist_784",version=1,return_X_y=True)
    X, y = data
    df = X
    df["label"] = y

    if 1:
        b = sns.countplot(x="label", data=df)
        plt.show()

    kf = model_selection.StratifiedKFold(n_splits=5, shuffle=True)

    df["kfold"] = -1
    for fold ,(trn_, val_) in enumerate(kf.split(X=df.drop(columns=["label"]).values,y=df["label"].values)):
        df.loc[val_, "kfold"] = fold
    
    if 1:
        print(df)

    if 1:
        for k in range(5):
            b = sns.countplot(x="label", data=df[df["kfold"]==k])
            plt.show()
    
    df.to_csv("../input/mnist_train_folds.csv")