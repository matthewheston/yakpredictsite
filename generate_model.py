from sklearn.linear_model import ElasticNet
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import pandas as pd
from sklearn.externals import joblib

# load up some yaks
yaks = pd.read_csv("data/northwestern_yaks.csv", escapechar="\\")

y = yaks["score"]

messages = yaks["message"]

vectorizer = CountVectorizer(binary=True)

X = vectorizer.fit_transform(messages)

enet = ElasticNet(alpha=0.1).fit(X, y)

model = Pipeline([('vect', vectorizer),
                    ('clf', enet)])

joblib.dump(model, "model.pkl")
