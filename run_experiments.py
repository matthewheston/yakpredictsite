from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.cross_validation import cross_val_score, KFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.grid_search import GridSearchCV
import pandas as pd
import numpy as np
import sys

yaks = pd.read_csv("northwestern_yaks_partitioned.csv", escapechar="\\", quotechar='"', error_bad_lines=False)
yaks['day_of_week'] = yaks['day_of_week'].astype('category')

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
    c_vectorizer = CountVectorizer(binary=True, ngram_range=(1,ngrams)).build_analyzer()

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
    logit_clf = LogisticRegression(penalty='l2', C=0.1, class_weight="auto")
    report_model(logit_clf, "Logistic Regression with l2 penalization")
    report_model(logit_clf, "Logistic Regression with l2 penalization bigrams", ngrams=2)
    report_model(logit_clf, "Logistic Regression with l2 penalization, with features", ngrams=2, features_to_include = temporal_features + text_features)
    nb_clf = MultinomialNB()
    report_model(nb_clf, "Naive Bayes")
    report_model(nb_clf, "Naive Bayes bigrams", ngrams=2)
    report_model(nb_clf, "Naive Bayes with features", ngrams=2, features_to_include = temporal_features + text_features)

def grid_search():
    discretize_scores()

    # set up grid search
    parameters = [{"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1.0, 100.0]}]
    clf = GridSearchCV(LogisticRegression(class_weight="auto"), parameters, "f1", cv=5, n_jobs = int(sys.argv[2])) # get number of parallel jobs from command line args

    # create feature matrix
    temporal_features = ['time_of_day', 'day_of_week']
    text_features = ['message_len', 'num_words', 'mean_word_len']
    features_to_include = temporal_features + text_features
    y = yaks["score"].values.astype(np.float32)
    vect = DictVectorizer()
    c_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2)).build_analyzer()
    temp = zip(*[list(yaks[x]) for x in ['message'] + features_to_include])
    tokenized = map(lambda x: get_features(x, c_vectorizer, features_to_include), temp)
    X = vect.fit_transform(tokenized)

    # run it and print results
    clf.fit(X, y)
    print "Best parameters set found on development set:"
    print
    print clf.best_estimator_
    print
    print "Grid scores on development set:"
    print
    for params, mean_score, scores in clf.grid_scores_:
        print "%0.3f (+/-%0.03f) for %r" % (
            mean_score, scores.std() / 2, params)

def print_top_results():
    discretize_scores()

    # create feature matrix
    temporal_features = ['time_of_day', 'day_of_week']
    text_features = ['message_len', 'num_words', 'mean_word_len']
    features_to_include = temporal_features + text_features
    y = yaks["score"].values.astype(np.float32)
    vect = DictVectorizer()
    c_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2)).build_analyzer()
    temp = zip(*[list(yaks[x]) for x in ['message'] + features_to_include])
    tokenized = map(lambda x: get_features(x, c_vectorizer, features_to_include), temp)
    X = vect.fit_transform(tokenized)

    clf = LogisticRegression(C=0.1, class_weight='auto', dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    clf.fit(X, y)

    feature_names = vect.get_feature_names()
    for i, class_label in enumerate([-1, 0, 1]):
        top20 = np.argsort(clf.coef_[i])[-20:]
        print("%s: %s" % (class_label,
              "\t\n ".join(feature_names[j] for j in top20)))

def test_unseen():
    unseen_yaks = pd.read_csv("fsu_heldout_partitioned.csv", escapechar="\\", quotechar='"', error_bad_lines=False)

    unseen_yaks.loc[unseen_yaks["score"] < 0, "score"] = -1
    unseen_yaks.loc[(unseen_yaks["score"] > 0) &
            (unseen_yaks["score"] < yaks["score"].quantile(.9)), "score"] = 0
    unseen_yaks.loc[unseen_yaks["score"] >= yaks["score"].quantile(.9), "score"] = 1

    discretize_scores()

    # create feature matrix
    temporal_features = ['time_of_day', 'day_of_week']
    text_features = ['message_len', 'num_words', 'mean_word_len']
    features_to_include = temporal_features + text_features
    y = yaks["score"].values.astype(np.float32)
    vect = DictVectorizer()
    c_vectorizer = CountVectorizer(binary=True, ngram_range=(1,2)).build_analyzer()
    temp = zip(*[list(yaks[x]) for x in ['message'] + features_to_include])
    tokenized = map(lambda x: get_features(x, c_vectorizer, features_to_include), temp)
    X = vect.fit_transform(tokenized)

    clf = LogisticRegression(C=0.1, class_weight='auto', dual=False, fit_intercept=True,
          intercept_scaling=1, penalty='l2', random_state=None, tol=0.0001)

    clf.fit(X, y)

    temp = zip(*[list(unseen_yaks[x]) for x in ['message'] + features_to_include])
    tokenized = map(lambda x: get_features(x, c_vectorizer, features_to_include), temp)
    X = vect.transform(tokenized)

    predicted = clf.predict(X)
    true = unseen_yaks.score

    f1 = f1_score(true, predicted, average="weighted")

    print "F1 score: %.5f" % f1



if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    if len(sys.argv) > 1:
        if sys.argv[1] == 'gridsearch':
            grid_search()
        if sys.argv[1] == 'features':
            print_top_results()
        if sys.argv[1] == 'test':
            test_unseen()
