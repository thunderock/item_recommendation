from os import path
import argparse

import sys
root_directory_path = path.abspath('')
sys.path.insert(0, root_directory_path)
import data_frame as df
import utilities as ut

parser = argparse.ArgumentParser()

csv_path = root_directory_path + '/movielens/'
parser.add_argument("-ds", "--data_size", type=str, default='1m', help="data set size (latest 2.5 mn)")

args = parser.parse_args()

if args.data_size == '1m':
    csv_path += 'ml-latest-small/ratings.csv'
    print(csv_path)
elif args.data_size == 'latest':
    csv_path += 'ml-latest/ratings.csv'
    pass
else:
    #probably need to see which more data sets can be used.
    args.data_size = '1m'
    csv_path += 'ml-latest-small/ratings.csv'
    print ("Wrong parameter for size of data points. Running with default, which is small 1m.\n")

data_frame = df.load_dataframe_from_file_path(csv_path)

ut.put_data_frame_in_db(data_frame)
