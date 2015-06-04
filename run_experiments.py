from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
import pandas as pd
import numpy as np

yaks = pd.read_csv("data/northwestern_yaks_partitioned.csv", escapechar="\\")

messages = yaks["message"]

def discretize_scores(quantile_score = .9):
    yaks.loc[yaks["score"] < 0, "score"] = -1
    yaks.loc[(yaks["score"] > 0) &
            (yaks["score"] < yaks["score"].quantile(quantile_score)), "score"] = 0
    yaks.loc[yaks["score"] >= yaks["score"].quantile(quantile_score), "score"] = 1
    print "Class distribution: -1 (%s) 0 (%s) 1 (%s)" % (len(yaks[yaks["score"] == -1]), len(yaks[yaks["score"] == 0]), len(yaks[yaks["score"] == 1]))
    print "\n\n\n"

def get_features(yak, vectorizer, feature_names):
    # Get all of the terms
    terms = vectorizer(yak[0])
    # Get the rest of the features
    d = {feature_names[i]: yak[i + 1] for i in range(len(feature_names))}
    # Add the terms to the dictionary
    for t in terms:
        d[t] = d.get(t, 0) + 1
    return d

def report_model(clf, clfname, ngrams=1, features_to_include = []):
    y = yaks["score"].values.astype(np.float32)
    vect = DictVectorizer()
    c_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2)).build_tokenizer()

    # Create temporary version of features that can be passed to get_features
    temp = zip(*[list(yaks[x]) for x in ['message'] + features_to_include])
    tokenized = map(lambda x: get_features(x, c_vectorizer, features_to_include), temp)
    X = vect.fit_transform(tokenized)
    print "Using %s. Bag of words, %s-grams, binary." % (clfname, ngrams)
    cross_val_f1 = cross_val_score(clf, X, y, cv=5, scoring='f1_weighted')
    cross_val_accuracy = cross_val_score(clf, X, y, cv=5, scoring='accuracy')
    print 'Average F1=%.5f (%.5f)' % (np.average(cross_val_f1), np.std(cross_val_f1))
    print 'Average accuracy=%.5f (%.5f)' % (np.average(cross_val_accuracy), np.std(cross_val_accuracy))
    print "\n\n\n"


def main():
    discretize_scores()
    temporal_features = ['time_of_day', 'day_of_week']
    text_features = ['message_len', 'num_words', 'mean_word_len']
    logit_clf = LogisticRegression(penalty='l2', class_weight="auto")
    report_model(logit_clf, "Logistic Regression with l2 penalization")
    report_model(logit_clf, "Logistic Regression with l2 penalization", ngrams=2)
    report_model(logit_clf, "Logistic Regression with l2 penalization, with features", ngrams=2, features_to_include = temporal_features + text_features)
    nb_clf = MultinomialNB()
    report_model(nb_clf, "Naive Bayes")
    report_model(nb_clf, "Naive Bayes", ngrams=2)
    report_model(nb_clf, "Naive Bayes with features", ngrams=2, features_to_include = temporal_features + text_features)

if __name__ == "__main__":
    main()
