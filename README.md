Tensorflow based Deals Recommendation System

Requirements:
Python >= 2.7.13
Tensorflow >= 1.2.1
Numpy >= 1.13.1
Pandas >= 0.20.3
Sklearn >= 0.18.2

Please run checking_installation to check if your versions are up to date.

Algorithms used:
k-truncated SVD un-supervised learning

Dataset Used:
MovieLens 2.5M dataset

Filtering Steps:
1. filtering users who have purchased less than 20 deals.

Assumptions:
1. Choices made by a user is independent of their age, gender and location but do depend on class of deals he has been 'interested' in (which is derived from his history).

