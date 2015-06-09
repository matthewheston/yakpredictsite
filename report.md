---
title: Report: Predicting Yak Scores based on text and temporal features
...

EECS 349, Northwestern University, Spring 2015

Matthew Heston
<heston@u.northwestern.edu>

Jeremy Foote
<jdfoote@u.northwestern.edu>

Task Overview
---------------

Our task, as [explained](abstract.html), was to predict the score that a yak posted on Yik Yak would receive. 

Data Collection
----------------

Data was collected from September 2014 to February 2015. A script ran every 1
hour and retrieved the last 100 yaks posted to 35 different campuses. In the
experiments and analysis listed below, we use data from only two of these
campuses: Northwestern University and Florida State University. Northwestern was
chosen as we are affiliated with the university and assumed we therefore might
have some intuition about why certain features might be associated with
different scores. Florida State was chosen as it by far produced the most yaks
of all the campuses in our collection, and we thought it would be valuable to
test our methods using a campus with a large number of documents for training.
For each yak collected, we have the message text, the time it was posted, and
the score of the yak as of the last time our script collected data about the
yak.

For our initial experiments, we had 26,628 yaks from Northwestern, and 112,518
yaks from FSU. Our cross validation results use these yaks. In addition, we have
7687 heldout Northwestern yaks and 33,584 heldout yaks FSU yaks for testing on
unseen data.

Feature Extraction
----------------

Our primary feature set consisted of bag of words representations of the yaks
themselves. We trained models using both unigram and bigram representations. In
both cases, we use a binary bag of words representation, in which each term is
represented by a 0 or 1 indicating whether or not it appeared in the yak.

In addition bag of words, we derived other features. These include features
based on the timestamp of the yak, including time of day (late night, early
morning, etc) and day of week. We also included addition features about the text
itself, including how many words were including the total number of characters
in the yak, the number of words in the yak, and the mean word length.


Learning Algorithms
-------------------

We trained both a logistic regression and a naive Bayes classifier using various
configurations of the features listed above. We used the implementations of
these algorithms in the Python library
[scikit-learn](http://scikit-learn.org/).

Hyperparameter tuning was performed for logistic regression using the
scikit-learn `gridsearch` module. Based on results from 5-fold cross-validation,
selecting parameters based on F1 score, we decided on using L2 penalization in
our logistic regression model and set the C value (inverse of regulization
strength) to 0.1. It should be noted that all models in our parameter tuning
performed fairly similarly, but this model did boost mean F1 score around 4%
from the lowest performing configuration. We used default parameter settings for
naive Bayes.

Final Results
-----------------

Results from 5-fold cross validation for each classifier, using different
features, are shown in the following tables. The first table shows results for
our corpus of Northwestern yaks. The second shows results for our corpus of
Florida State yaks.

<table>
  <tr>
    <th></th>
    <th>Unigram Features</th>
    <th>Unigram + Bigram Features</th>
    <th>Unigram + Bigram + Time and Text Features</th>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>68.6%</td>
    <td>68.6%</td>
    <td>68.1%</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>68.9%</td>
    <td>69.2%</td>
    <td>69.2%</td>
  </tr>
  <tr>
    <td colspan="4">Note: Percentages reflect average F1 score from 5-fold cross validation.</td>
  </tr>
</table>

<table>
  <tr>
    <th></th>
    <th>Unigram Features</th>
    <th>Unigram + Bigram Features</th>
    <th>Unigram + Bigram + Time and Text Features</th>
  </tr>
  <tr>
    <td>Naive Bayes</td>
    <td>75.2%</td>
    <td>75.1%</td>
    <td>75.3%</td>
  </tr>
  <tr>
    <td>Logistic Regression</td>
    <td>76.0%</td>
    <td>76.0%</td>
    <td>76.1%</td>
  </tr>
  <tr>
    <td colspan="4">Note: Percentages reflect average F1 score from 5-fold cross validation.</td>
  </tr>
</table>

In both cases, both classifiers perform comparably. All of our feature sets also
perform comparably. Adding bigrams does not improve results significantly over
using only unigrams. Nor do our derived time and text features make a
significant difference. We see better F1 scores with the classifiers trained on
Florida State corpus, likely due to the relatively much larger collection of
documents for training.
