---
title: Abstract - Predicting Yak Scores based on text and temporal features
...

EECS 349, Northwestern University, Spring 2015

Matthew Heston
<heston@u.northwestern.edu>

Jeremy Foote
<jdfoote@u.northwestern.edu>

Synopsis
----------

### Context ###
Yik Yak is an anonymous, location-aware
mobile application in which users can create short posts, and can view and
upvote or downvote posts created near their location. While anonymous,
location-based messages have been a part of the world for a long time (e.g.,
writing on the bathroom stall), the particular combination of anonymity and mass
social feedback is enabled by GPS-enabled smartphones. Our goal is to understand
how people are using this new "place", and whether the norms and uses differ
by location.

### Data and Analysis ###
We captured approximately 200,000 "yaks" from Northwestern University campus and Florida State University campus. For each yak, we capture the message, the approximate number of votes that it received, and the timestamp from when it was posted. Our analyses are conducted separately for each campus, and results are compared.

We use logistic regression and naive bayes learners, starting with a "bag of words" approach to the text, and then adding temporal and higher-level text features (such as post length and mean word length). We held out 5,000 yaks from each campus, and then trained on the rest, using 5-fold cross-validation. We also tested logistic regression classifiers on held out data for each of the two campuses. In all cases, we report F1 score, given the imbalanced nature of our class distribution

### Findings ###
Adding both bigram and our higher level features to a basic unigram bag of words model in most cases slightly increased classifier performance, but not by that much. Logistic regression outperformed Naive Bayes for each campus. We achieved between 70% and 75% F1 scores. Inspecting the top features for each class for each campus allows us to start to think about what types of words and phrases are awarded and punished on Yik Yak. While interpretation of all these terms is difficult, results suggest local references tend to do well while offensive language tends not to.
