from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.externals import joblib

# load up some yaks
yaks = pd.read_csv("data/northwestern_yaks.csv", escapechar="\\")


def discretize_scores(quantile_score): 
    yaks.loc[yaks["score"] < 0, "score"] = -1 
    yaks.loc[(yaks["score"] > 0) & (yaks["score"] < yaks.score.quantile(quantile_score)), "score"] = 0 
    yaks.loc[yaks["score"] >= yaks.score.quantile(quantile_score), "score"] = 1 

discretize_scores(.9)

y = yaks["score"]

messages = yaks["message"]

vectorizer = CountVectorizer(binary=True)

X = vectorizer.fit_transform(messages)

clf = LogisticRegression().fit(X, y)

model = Pipeline([('vect', vectorizer),
                    ('clf', clf)])

joblib.dump(model, "model.pkl")
