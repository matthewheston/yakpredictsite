---
title: Predicting Yak Scores based on text and temporal features
...

EECS 349, Northwestern University, Spring 2015

Matthew Heston
<matthewheston@gmail.com>

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

We use logistic regression and naive bayes learners, starting with a "bag of words" approach to the text, and then adding temporal and higher-level text features (such as post length and mean word length). We held out 5,000 yaks from each campus, and then trained on the rest, using 5-fold cross-validation. We report F1 performance on both the training and the test set.

### Findings ###
