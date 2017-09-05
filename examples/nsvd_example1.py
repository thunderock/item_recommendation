
from os import path
import numpy as np
import argparse

import sys
root_directory_path = path.abspath('')
sys.path.insert(0, root_directory_path)
import data_frame as df
import svd_model as re
import pandas as pd

parser = argparse.ArgumentParser()


csv_path = root_directory_path + '/movielens/'
parser.add_argument("-b", "--batch", type=int, default=700, help="batch size (700)")
parser.add_argument("-i", "--iterations", type=int, default=10000, help="iterations (1000)")
parser.add_argument("-d", "--dimensions", type=int, default=12, help="vector dimensions (12)")
parser.add_argument("-ds", "--data_size", type=str, default='1m', help="data set size (latest 2.5 mn)")
parser.add_argument("-m", "--model", type=str, default='svd', help="svd or nsvd (default=svd)")
parser.add_argument("-S", "--nsvd_size", type=str, default='mean', help="size of vectors of nsvd model max, mean or min")
parser.add_argument("-I", "--info", type=str, default='True', help="printing training info")
parser.add_argument("-M", "--momentum", type=float, default=0.926, help="momentum factor")
parser.add_argument("-l", "--learning", type=float, default=0.001, help="Learning rate")
parser.add_argument("-r", "--reg", type=float, default=0.0003, help="regularizer const for loss function")
parser.add_argument("-uid", "--user_id", type=int, default=671, help="user_id for which prediction is required to be made (default = 671)")
parser.add_argument("-u", "--use", type=str, default='csv', help="graph or csv")

args = parser.parse_args()

if args.data_size == '1m':
    csv_path += 'ml-latest-small/ratings.csv'
elif args.data_size == 'latest':
    csv_path += 'ml-latest/ratings.csv'
    pass
else:
    #probably need to see which more data sets can be used.
    args.data_size = 'latest'
    csv_path += 'ml-latest/ratings.csv'
    print ("Wrong parameter for size of data points. Running with default, which is latest.\n")

if args.use == 'graph':
  data_frame = df.load_data_from_db()
else:
  data_frame = df.load_dataframe_from_file_path(csv_path)

#ut.put_data_frame_in_db(data_frame)
user_queried = args.user_id
if args.model == "svd":
    model = re.SVDmodel(user_queried, data_frame, 'user', 'deal', 'rating')
else:
    model = re.SVDmodel(user_queried, data_frame, 'user', 'deal', 'rating', args.model, args.nsvd_size)
if len(model.valid) != 0:
    dimension = args.dimensions
    regularizer_constant = args.reg
    learning_rate = args.learning
    batch_size = args.batch
    num_steps = args.iterations
    momentum_factor = args.momentum
    info = True if args.info else False
    model.training(dimension, regularizer_constant, learning_rate, momentum_factor, batch_size, num_steps, info)
    prediction = model.valid_prediction()
    print("\nThe mean square error of the whole valid dataset is ", prediction)
    user_example = np.array(model.valid['user'])
    deals_example = np.array(model.valid['deal'])
    actual_ratings = np.array(model.valid['rating'])
    predicted_ratings = model.prediction(user_example, deals_example)
    print("\nuser:             \n")
    print(user_example[0])
    result = pd.DataFrame({'deal_ids': deals_example, 'predicted_rating': predicted_ratings, 'original_ratings': actual_ratings})
    print(result)
    #print("\ndeals:             \n")
    #print(deals_example)
    #print("\npredicted ratings: \n")
    #print(predicted_ratings)
    #print("\nAnd in reality the scores are:")
    #print(actual_ratings)






