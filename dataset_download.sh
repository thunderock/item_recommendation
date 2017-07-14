#!/usr/bin/env bash

DIR=$(pwd)/movielens
mkdir -p ${DIR}
#getting dataset zip from movielens
wget http://files.grouplens.org/datasets/movielens/ml-latest.zip -O ${DIR}/ml-latest.zip
unzip ${DIR}/ml-latest.zip -d ${DIR}

wget http://files.grouplens.org/datasets/movielens/ml-latest-small.zip -O ${DIR}/ml-latest-small.zip
unzip ${DIR}/ml-latest-small.zip -d ${DIR}

rm movielens/ml-latest.zip
rm movielens/ml-latest-small.zip