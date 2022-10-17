import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
from sklearn import metrics, preprocessing
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import torch.nn.functional as F


class EntityEmbed(nn.Module):
    def __init__(self, df: pd.DataFrame, catcols: list):
        super().__init__()

        self.catcols = catcols

        embed_dim_sum = 0
        self.embed = {}
        self.drop1 = {}
        for col in catcols:
            num_unique_values = int(df[col].nunique())
            embed_dim = min((num_unique_values + 1) // 2, 50)
            embed_dim_sum += embed_dim
            self.embed[col] = nn.Embedding(num_unique_values, embed_dim)
            self.drop1[col] = nn.Dropout(p=0.3)

        self.bn1 = nn.BatchNorm1d(embed_dim_sum)

        self.fc2 = nn.Linear(embed_dim_sum, 300)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(p=0.3)
        self.bn2 = nn.BatchNorm1d(300)

        self.fc3 = nn.Linear(300, 300)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(p=0.3)
        self.bn3 = nn.BatchNorm1d(300)

        self.fc4 = nn.Linear(300, 1)

    def forward(self, x: dict):

        embeds = {}
        for col in self.catcols:
            embeds[col] = self.embed[col](x[col])
            embeds[col] = self.drop1[col](embeds[col])

        out = torch.cat([embeds[col] for col in self.catcols], dim=1)

        out = self.fc2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        out = self.bn2(out)

        out = self.fc3(out)
        out = self.relu3(out)
        out = self.drop3(out)
        out = self.bn3(out)

        out = self.fc4(out)

        return out


class EntityEmbedDataset(Dataset):
    def __init__(self, df: pd.DataFrame, catcols: list):
        super().__init__()

        self.len = len(df)
        self.catcols = catcols
        self.y = df["target"].values
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
    df = pd.read_csv("../input/train_folds.csv")

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

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(20):

        model.train()
        train_loss = 0
        for _, (x, y) in enumerate(tqdm(dl_train, total=int(len(dl_train)))):
            b = len(y)
            y = torch.unsqueeze(y, 1)
            optimizer.zero_grad()
            outputs = model(x)
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
                b = len(y)
                outputs = model(x)
                y = y.detach().numpy()
                outputs = outputs.detach().numpy()
                valid_loss += metrics.roc_auc_score(y, outputs) * b
            valid_loss /= len(ds_valid)
            print(valid_loss)
