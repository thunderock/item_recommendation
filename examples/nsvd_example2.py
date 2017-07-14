
from os import path
import numpy as np
import argparse
import pandas as pd

import sys
root_directory_path = path.abspath('..')
sys.path.insert(0, root_directory_path)
import data_frame as df
import svd_model as re
import utilities as ut

parser = argparse.ArgumentParser()


csv_path = root_directory_path + '/movielens/'
parser.add_argument("-b", "--batch", type=int, default=700, help="batch size (700)")
parser.add_argument("-i", "--iterations", type=int, default=10000, help="iterations (1000)")
parser.add_argument("-d", "--dimensions", type=int, default=12, help="vector dimensions (12)")
parser.add_argument("-ds", "--data_size", type=str, default='1m', help="data set size (latest 2.5 mn)")
parser.add_argument("-m", "--model", type=str, default='nsvd', help="svd or nsvd (default=svd)")
parser.add_argument("-S", "--nsvd_size", type=str, default='mean', help="size of vectors of nsvd model max, mean or min")
parser.add_argument("-I", "--info", type=str, default='True', help="printing training info")
parser.add_argument("-M", "--momentum", type=float, default=0.926, help="momentum factor")
parser.add_argument("-l", "--learning", type=float, default=0.001, help="Learning rate")
parser.add_argument("-r", "--reg", type=float, default=0.0003, help="regularizer const for loss function")
parser.add_argument("-uid", "--user_id", type=int, default=671, help="user_id for which prediction is required to be made (default = 671)")


args = parser.parse_args()

if args.data_size == '1m':
    csv_path += 'ml-latest-small/ratings.csv'
elif args.data_size == 'latest':
    csv_path += 'ml-latest/ratings.csv'
    pass
else:
    args.data_size = 'latest'
    csv_path += 'ml-latest/ratings.csv'
    print ("Wrong parameter for size of data points. Running with default, which is latest.\n")

data_frame = df.load_dataframe_from_file_path(csv_path)
#data_frame = df.load_data_from_db()
#ut.put_data_frame_in_db(data_frame)
user_queried = args.user_id
is_future_prediction = True
if args.model == "svd":
    model = re.SVDmodel(user_queried, data_frame, 'user', 'deal', 'rating')
else:
    model = re.SVDmodel(user_queried, data_frame, 'user', 'deal', 'rating', args.model, args.nsvd_size, is_future_prediction)
if len(model.train) != 0:
    assert len(model.valid) == 0
    dimension = args.dimensions
    regularizer_constant = args.reg
    learning_rate = args.learning
    batch_size = args.batch
    num_steps = args.iterations
    momentum_factor = args.momentum
    info = True if args.info else False
    model.training(dimension, regularizer_constant, learning_rate, momentum_factor, batch_size, num_steps, info)
    deals_example = ut.get_deals_not_rated_by_this_customer(data_frame, 'user', 'deal', user_queried)
    user_example = np.array(ut.get_np_redundant_array(len(deals_example), user_queried))
    predicted_ratings = model.prediction(user_example, deals_example)
    result = pd.DataFrame({'rating': predicted_ratings, 'deal': deals_example})
    result = result[result['rating'] < 5]
    print("Total number of deals which are not rated : ")
    print(len(result))
    print ("average of all the ratings")
    print(np.average(predicted_ratings))
    result = result.sort_values('rating', ascending=[False])
    print("\nPredicted ratings for deals for ")
    print (user_queried)
    print("\nTop 20 relevant deals for this user:             \n")
    print(result.head(20))





