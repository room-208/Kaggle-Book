from os import pread
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
from sklearn import linear_model, metrics, model_selection

import nltk

nltk.download("punkt")

from nltk.tokenize import word_tokenize


def find_sentiment(sentence: str, pos: set, neg: set):
    sentence - sentence.split()
    settence = set(sentence)
    num_common_pos = len(sentence.intersection(pos))
    num_common_neg = len(sentence.intersection(neg))

    if num_common_pos > num_common_neg:
        return "positive"
    else:
        return "negative"
    return "neutral"


if __name__ == "__main__":
    if 0:
        sentence = "hi, how are you?"
        print(sentence.split())
        print(word_tokenize(sentence))

    if 0:
        corpus = [
            "Probably my all-time favorite movie",
            "a story of selflessness story",
            "sacrifice and dedication to a noble cause",
            "story",
        ]

        ctv = CountVectorizer()
        ctv.fit(corpus)
        corpus_transformed = ctv.transform(corpus)
        print(corpus_transformed)
        print(ctv.vocabulary_)

    if 1:
        df = pd.read_csv("../input/IMDBDataset.csv")

        df["sentiment"] = df["sentiment"].apply(lambda x: 1 if x == "positive" else 0)

        df["kfold"] = -1

        kf = model_selection.StratifiedKFold(n_splits=5)

        y = df["sentiment"].values

        for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, "kfold"] = f

        for fold in range(5):
            train_df = df[df["kfold"] != fold].reset_index(drop=True)
            test_df = df[df["kfold"] == fold].reset_index(drop=True)

            count_vec = CountVectorizer(tokenizer=word_tokenize, token_pattern=None)

            count_vec.fit(train_df["review"])

            print(len(count_vec.vocabulary_))

            X_train = count_vec.transform(train_df["review"])
            X_test = count_vec.transform(test_df["review"])

            model = linear_model.LogisticRegression()

            model.fit(X_train, train_df["sentiment"])

            preds = model.predict(X_test)

            acc = metrics.accuracy_score(test_df["sentiment"], preds)

            print(fold, acc)
