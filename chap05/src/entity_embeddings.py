import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import metrics, preprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np


def reduce_mem_usage(df, verbose=True):
    """
    Reduce file memory usage
    Source: https://www.kaggle.com/artgor

    Parameters:
    -----------
    df: DataFrame
        Dataset on which to perform transformation
    verbose: bool
        Print additional information
    Returns:
    --------
    DataFrame
        Dataset as pandas DataFrame
    """
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                c_prec = df[col].apply(lambda x: np.finfo(x).precision).max()
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                    and c_prec == np.finfo(np.float16).precision
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                    and c_prec == np.finfo(np.float32).precision
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )

    return df


class EntityEmbed(nn.Module):
    def __init__(self, df: pd.DataFrame, catcols: list):
        super().__init__()

        self.catcols = catcols

        embed_dim_sum = 0
        self.embed_lists = nn.ModuleList()
        for col in catcols:
            num_unique_values = int(df[col].nunique())
            embed_dim = min((num_unique_values + 1) // 2, 50)
            embed_dim_sum += embed_dim
            self.embed_lists.append(nn.Embedding(num_unique_values, embed_dim))

        self.bn1 = nn.BatchNorm1d(embed_dim_sum)
        self.drop1 = nn.Dropout(p=0.3)

        self.fc2 = nn.Linear(embed_dim_sum, 300)
        nn.init.kaiming_normal_(self.fc2.weight.data)
        self.drop2 = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm1d(300)

        self.fc3 = nn.Linear(300, 300)
        nn.init.kaiming_normal_(self.fc3.weight.data)
        self.drop3 = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(300)

        self.fc4 = nn.Linear(300, 2)
        nn.init.kaiming_normal_(self.fc4.weight.data)

        self.softmax = nn.Softmax(dim=1)

    def forward(self, x: dict):

        out = torch.cat(
            [l(x[col]) for l, col in zip(self.embed_lists, self.catcols)], dim=1
        )
        out = self.bn1(out)
        out = self.drop1(out)

        out = F.relu(self.fc2(out))
        out = self.drop2(out)
        out = self.bn2(out)

        out = F.relu(self.fc3(out))
        out = self.drop3(out)
        out = self.bn3(out)

        out = self.fc4(out)
        out = self.softmax(out)

        return out


class EntityEmbedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, catcols: list):
        super().__init__()
        self.len = len(df)
        self.catcols = catcols
        self.y = df["target"].values.reshape(-1, 1)
        self.X = {}
        for col in self.catcols:
            self.X[col] = df[col].values

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        x = {}
        for col in self.catcols:
            x[col] = self.X[col][index]
        return x, self.y[index]


if __name__ == "__main__":
    df = pd.read_csv("./chap05/input/train_folds.csv")
    df = reduce_mem_usage(df)

    features = [f for f in df.columns if f not in ["id", "target", "kfold"]]

    for col in features:
        lbl_enc = preprocessing.LabelEncoder()
        df[col] = df[col].astype(str).fillna("NONE")
        df[col] = lbl_enc.fit_transform(df[col].values)

    df_train = df[df["kfold"] != 0].reset_index(drop=True)
    df_valid = df[df["kfold"] == 0].reset_index(drop=True)

    model = EntityEmbed(df, features)

    ds_train = EntityEmbedDataset(df_train, features)
    ds_valid = EntityEmbedDataset(df_valid, features)
    dl_train = DataLoader(
        ds_train,
        batch_size=1024,
        shuffle=True,
        num_workers=2,
        drop_last=True,
        pin_memory=True,
    )
    dl_valid = DataLoader(
        ds_valid,
        batch_size=1024,
        num_workers=2,
        pin_memory=True,
    )

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=9e-3)

    for epoch in range(20):

        model.train()
        train_loss = 0
        for _, (x, y) in enumerate(tqdm(dl_train, total=int(len(dl_train)))):
            b = y.shape[0]
            optimizer.zero_grad()
            outputs = model(x)[:, 1].reshape(-1, 1)
            loss = criterion(outputs, y.float())
            loss.backward()
            optimizer.step()
            train_loss += float(loss) * b
        train_loss /= len(ds_train)
        print(train_loss)

        with torch.no_grad():
            model.eval()
            valid_loss = 0
            for _, (x, y) in enumerate(tqdm(dl_valid, total=int(len(dl_valid)))):
                b = y.shape[0]
                outputs = model(x)[:, 1].reshape(-1, 1)
                y = y.numpy()
                outputs = outputs.numpy()
                valid_loss += metrics.roc_auc_score(y, outputs) * b
            valid_loss /= len(ds_valid)
            print(valid_loss)
