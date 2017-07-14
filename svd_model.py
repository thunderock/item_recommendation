#!/usr/bin/env python3

import numpy as np
import data_frame
import pandas as pd
import tensor_flow_models as tf_models
from py2neo import Graph, Node, Relationship


class SVDmodel(object):
    def __init__(self,
                 user_queried,
                 df,
                 users,
                 items,
                 ratings,
                 model='svd',
                 nsvd_size='mean',
                 if_no_validation=False):
        self.df = df
        self.if_no_validation = if_no_validation
        self.users = users
        self.items = items
        self.ratings = ratings
        self.model = model
        self.size = len(df)
        self.num_of_users = max(self.df[self.users]) + 1
        self.num_of_items = max(self.df[self.items]) + 1
        self.train, self.test, self.valid = self.data_separation(user_queried)
        if model == 'nsvd':
            self.finder = data_frame.DealExtractor(df, self.users,
                                                 self.items,
                                                 self.ratings, nsvd_size)

    #def get_user_rated_deals(self, user_queried):
    #    query = """
    #    MATCH (u:User) - [:rates] -> (d:Deal)
    #    WHERE u.id = {user_id}
    #    RETURN d
    #    """
    #    graph = Graph(password="cyclops")
    #    graph.run(query, user_id=user_queried)

    def get_data_frames_with_and_without_this_user(self, user):
        return self.df[self.df[self.users] != user], self.df[self.df[self.users] == user]

    def data_separation_for_no_validation(self, user):
        """
        test_data is empty
        validation is 5%
        training is 95%
        :param user:
        :return:
        """
        df_validation = pd.DataFrame(columns=['user', 'deal', 'rating'])
        rows = len(self.df)
        random_ids = np.random.permutation(rows)
        split_index = int(rows * 0.95)
        random_df = self.df.iloc[random_ids].reset_index(drop=True)
        df_train = random_df[0:split_index]
        df_test = random_df[split_index:].reset_index(drop=True)
        return df_train, df_test, df_validation

    def data_separation(self, user_queried):
        if self.if_no_validation:
            return self.data_separation_for_no_validation(user_queried)
        all_users_df, this_user_df = self.get_data_frames_with_and_without_this_user(user_queried)
        user_rows = len(this_user_df)
        if user_rows < 20:
            print ("This user has less than 20 deals rated. Sorry we can't help here.")
            return [], [], []
        random_user_ids = np.random.permutation(user_rows)
        random_user_df = this_user_df.iloc[random_user_ids].reset_index(drop=True)
        split_user_index = int(user_rows * 0.25)
        df_validation = random_user_df[0:split_user_index]
        df = pd.concat([all_users_df, this_user_df[split_user_index:].reset_index(drop=True)])
        rows = len(df)
        random_ids = np.random.permutation(rows)
        random_df = df.iloc[random_ids].reset_index(drop=True)
        split_index = int(rows * 0.95)
        #new_split = split_index + int((rows - split_index) * 0.5)
        df_train = random_df[0:split_index]
        df_test = random_df[split_index:].reset_index(drop=True)
        #df_validation = random_df[new_split:].reset_index(drop=True)
        return df_train, df_test, df_validation

    def training(self,
                 hp_dim,
                 hp_reg,
                 learning_rate,
                 momentum_factor,
                 batch_size,
                 num_steps,
                 verbose=True):
        self.train_batches = data_frame.BatchGenerator(self.train,
                                                        batch_size,
                                                        self.users,
                                                        self.items,
                                                        self.ratings)

        self.test_batches = data_frame.BatchGenerator(self.test,
                                                       batch_size,
                                                       self.users,
                                                       self.items,
                                                       self.ratings)

        self.valid_batches = data_frame.BatchGenerator(self.valid,
                                                        len(self.valid),
                                                        self.users,
                                                        self.items,
                                                        self.ratings)
        if self.model == 'svd':
            self.tf_counterpart = tf_models.SVDTrainer(self.num_of_users,
                                                self.num_of_items,
                                                self.train_batches,
                                                self.test_batches,
                                                self.valid_batches)
        else:
            self.tf_counterpart = tf_models.SVDTrainer(self.num_of_users,
                                                self.num_of_items,
                                                self.train_batches,
                                                self.test_batches,
                                                self.valid_batches,
                                                self.finder,
                                                self.model)

        self.tf_counterpart.training(hp_dim,
                                     hp_reg,
                                     learning_rate,
                                     momentum_factor,
                                     num_steps,
                                     verbose)
        self.duration = round(self.tf_counterpart.general_duration, 2)
        if verbose:
            self.tf_counterpart.print_stats()

    def valid_prediction(self):
        return self.tf_counterpart.prediction(show_valid=True)

    def prediction(self, list_of_users, list_of_items):
        return self.tf_counterpart.prediction(list_of_users, list_of_items)
