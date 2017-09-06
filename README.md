Tensorflow based Deals Recommendation System

Requirements: Need to install these python libraries
Python3
Tensorflow
Numpy
Pandas
py2neo


backfilling data from dataset to db
if you are  using this for first time you need to change password to "cyclops" for code to work. Use this command for that
:server change-password
this will ask you to type in old password and new password. To get old default password go to this link:
https://github.com/neo4j/neo4j/issues/5444

run this command
python3 examples/data_fill.py
you can provide args to pick to pick different datasets

after backfill to run any example:
python3 filename

Algorithms used:
k-truncated SVD un-supervised learning

Dataset Used:
MovieLens 2.5M dataset

Filtering Steps:
1. filtering users who have purchased less than 20 deals.

Assumptions:
1. Choices made by a user is independent of their age, gender and location but do depend on class of deals he has been 'interested' in (which is derived from his history).

