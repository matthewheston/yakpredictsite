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

Our task, as [explained](abstract.html), was to predict the score that a yak
posted on Yik Yak would receive. Yaks that reach a score of -5 are automatically
deleted, making the lowest possible score -4. There is no upper limit on scores.
Rather than work directly with the continuous score value, we discretize scores
by using the following rules:

-  All yaks with a score of less than 0 form a category.
-  All yaks with scores higher than the 90th percentile form a category.
-  The rest of the yaks then form a middle category.

Our task is then to predict which of these three classes a yak will fall into.
These class distributions are generally not uniform, with most yaks falling into
the "middle" category, and much smaller percentage falling into the negative and
highly upvoted categories. We therefore use F1 as an evaluation metric
throughout this paper.

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

## Cross Validation

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

## Testing on Heldout Data

Training on all 26,628 Northwestern yaks and testing on our heldout 7,687 yaks
resulted in an F1 score of 76.7%. Doing the same with our 112,518 FSU yaks and
then testing on the heldout set of 33,584 yaks resulted in an F1 score of 74.7%.
In both of these cases, we use our logistic regression classifier and unigra,
bigram, and other other time and text features. The results here are comparable
to our cross validation results, though the Northwestern classifier performed
much better than expected given its cross validation results.

## Top Features for Each Classifier

Below we present the top 20 terms associated with each class, for both
Northwestern and Florida State.

**Florida State**

_Negative_
fartmonster  
chris  
lmfao  
john  
shrek  
grindr  
kik  
justin  
you if  
sunny  
jimmy  
bradsmithfsu  
the middle  
go gators  
salley  
doebotz  
nigga  
middle  
bit  
page  

_Middle_
it or  
pack  
bad  
summer  
away  
dirac  
that was  
basketball  
wishing  
that is  
traditions  
never had  
buddy  
to just  
girl to  
fml  
totally  
finding  
the one  
crazy  

_Highly Positive_
chiefs  
petition  
95  
password  
pregnant  
gator  
uf  
pregnancy  
years ago  
fsu  
strozier  
college  
upvote  
publix  
college where  
99  
unconquered  
cancer  
tallahassee  

**Northwestern**

_Negative_
anyone  
ass  
ugh  
women  
south  
zbt  
in norris  
cigs  
that the  
palestine  
smoke  
white  
wet  
suck  
how can  
me out  
israel  
cig  
norris  
cunt  

_Middle_
wash  
post  
music  
breakfast  
less  
what the  
the fuck  
forever  
the first  
could  
time  
mudd  
asleep  
outside  
hours  
beer  
the day  
you guys  
when your  
stress  

_Highly Upvoted_
different  
99  
kappa  
staying  
his  
chipotle  
petition  
is like  
single  
student  
wearing  
awkward  
college  
grade  
degree  
shark  
date  
degrees  
gpa  
northwestern  

Discussion
-----------

It is first interesting to note that the addition of bigrams and our other
derived features did not seem to boost performance that much in either of our
classifiers. (Why might this be?)

Inspecting our predictive features is not always intuitive, though it is in some
cases entertaining ("Chipotle" being highly predictive of being upvoted, for
example.) Some examples are intuitive and illustrative, however. We see that
certain offensive language is associated with being downvoted. School names or
mascots, on the other hand, are associated with a yak doing well. While it's
difficult to extrapolate too much from this, we can begin to see that in some
cases, students may reward local and relevant posts while punishing offensive
content, which is what we might expect. There are some surprising cases,
however. "Go gators" is predictive of being downvoted while "FSU" is associated
with upvotes. More qualitative analysis would be necessary to understand
possibly differences here.
