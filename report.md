---
title: Report: Predicting Yak Scores based on text and temporal features
...

EECS 349, Northwestern University, Spring 2015

Matthew Heston
<matthewheston@gmail.com>

Jeremy Foote
<jdfoote@u.northwestern.edu>

Task Overview
---------------

Our task, as [explained](abstract.html), was to predict the score that a yak posted on Yik Yak would receive. 

Data Collection
----------------

Feature Extraction
----------------

We began our task with a text-centric approach. We extracted the terms used in each post, and used a binary bag-of-words approach, where the existence of a term at least one time in the post is treated as a "1", and its absence as a "0".

However, there are other aspects of a post which may affect its popularity. We included both additional features of the text itself, and temporal features of the post. Specifically, we calculated the number of words, the mean word length, the time of day when the post was made, and the day of the week.

Learning Algorithms
-------------------

We focused on two algorithms for this task: logistic regression and Naive Bayes. For each, we used the algorithm as implemented in the Python library scikit-learn. 

Algorithm Tuning
----------------

Final Results
-----------------
