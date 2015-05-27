from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

yaks = pd.read_csv("data/northwestern_yaks.csv", escapechar="\\")

messages = yaks["message"]

def discretize_scores():
    yaks.loc[yaks["score"] < 0, "score"] = -1
    yaks.loc[(yaks["score"] > 0) & (yaks["score"] <= np.mean(yaks["score"]) + np.std(yaks["score"])), "score"] = 0
    yaks.loc[yaks["score"] >= np.mean(yaks["score"]) + np.std(yaks["score"]), "score"] = 1
    print "Class distribution: -1 (%s) 0 (%s) 1 (%s)" % (len(yaks[yaks["score"] == -1]), len(yaks[yaks["score"] == 0]), len(yaks[yaks["score"] == 1]))
    print "\n\n\n"

def report_unigrams_binary(clf, clfname):
    y = yaks["score"].values.astype(np.float32)
    vectorizer = CountVectorizer(binary=True)
    X = vectorizer.fit_transform(messages.tolist())
    print "Using %s. Bag of words, 1-gram, binary." % clfname
    cross_val_f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')
    cross_val_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print 'Average F1=%.5f (%.5f)' % (np.average(cross_val_f1), np.std(cross_val_f1))
    print 'Average accuracy=%.5f (%.5f)' % (np.average(cross_val_accuracy), np.std(cross_val_accuracy))
    print "\n\n\n"

def report_bigrams_binary(clf, clfname):
    y = yaks["score"].values.astype(np.float32)
    vectorizer = CountVectorizer(binary=True, ngram_range=(1,2))
    X = vectorizer.fit_transform(messages.tolist())
    print "Using %s. Bag of words, 2-gram, binary." % clfname
    cross_val_f1 = cross_val_score(clf, X, y, cv=5, scoring='f1')
    cross_val_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print 'Average F1=%.5f (%.5f)' % (np.average(cross_val_f1), np.std(cross_val_f1))
    print 'Average accuracy=%.5f (%.5f)' % (np.average(cross_val_accuracy), np.std(cross_val_accuracy))
    print "\n\n\n"

def main():
    discretize_scores()
    print yaks["score"].unique()
    logit_clf = LogisticRegression(penalty='l2', class_weight="auto")
    report_unigrams_binary(logit_clf, "Logistic Regression with l2 penalization")
    report_bigrams_binary(logit_clf, "Logistic Regression with l2 penalization")
    nb_clf = MultinomialNB()
    report_unigrams_binary(nb_clf, "Naive Bayes")
    report_bigrams_binary(nb_clf, "Naive Bayes")

if __name__ == "__main__":
    main()
