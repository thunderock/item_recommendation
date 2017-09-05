import numpy as np
import pandas as pd
from py2neo import Graph


def load_dataframe_from_file_path(path):
    raw_data_frame = pd.read_csv(path)
    raw_data_frame['userId'] = raw_data_frame['userId'] - 1
    raw_data_frame['dealId'] = raw_data_frame['dealId'] - 1
    raw_data_frame['rating'] = raw_data_frame['rating'].astype(np.float)
    raw_data_frame['user'] = raw_data_frame['userId'].astype(np.int32)
    raw_data_frame['deal'] = raw_data_frame['dealId'].astype(np.int32)
    return raw_data_frame[['user', 'deal', 'rating']]


def load_data_from_db():
    query = """
    MATCH (u:User)-[r:rates]->(d:Deal) 
    RETURN u.id, r.rating, d.id
    """
    data = Graph(password="cyclops").run(query)
    df = pd.DataFrame(columns=['user', 'deal', 'rating'])
    for row in data:
        df = df.append({'user': row['u.id'], 'deal': row['d.id'], 'rating': row['r.rating']}, ignore_index=True)
    df['rating'] = df['rating'].astype(np.float)
    df['user'] = df['user'].astype(np.int32)
    df['deal'] = df['deal'].astype(np.int32)
    return df

class DealExtractor(object):
    def __init__(self, df, users, deals, ratings, nsvd_size):
        self.users = users
        self.deals = deals
        self.df = df
        self.dic = {}
        self._set_deal_dictionary(nsvd_size)

    def get_items(self, user):
        user_df = self.df[self.df[self.users] == user]
        return np.array(user_df[self.deals])

    def _set_deal_dictionary(self, size_command="mean"):
        if not self.dic:
            all_users = self.df[self.users].unique()
            new_item = max(self.df[self.deals].unique()) + 1
            sizes = {}
            graph = Graph(password="cyclops")
            query = """
            MATCH (u:User) - [:rates] ->  (d:Deal)
            WHERE u.id = {user_id}
            RETURN d
            """
            print("\nWriting dictionary ...")
            for user in all_users:
                user_deals = graph.run(query, user_id=user.item())
                deals_rated = np.array([deal['d']['id'] for deal in user_deals])
                self.dic[user] = deals_rated
                sizes[user] = len(deals_rated)
            if size_command == "max":
                self.size = np.max(list(sizes.values()))
            elif size_command == "mean":
                self.size = int(np.mean(list(sizes.values())))
            elif size_command == "min":
                self.size = np.min(list(sizes.values()))
            print("Resizing ...")
            for user in all_users:
                if self.size <= sizes[user]:
                    self.dic[user] = self.dic[user][0:self.size]
                else:
                    difference_of_sizes = self.size - sizes[user]
                    tail = [new_item for i in range(difference_of_sizes)]
                    tail = np.array(tail)
                    result = np.concatenate((self.dic[user], tail), axis=0)
                    result = result.astype(np.int32)
                    self.dic[user] = result
            print("Generating size factors ...")
            if size_command == "max":
                for user in all_users:
                    sizes[user] = 1/np.sqrt(sizes[user])
                self.size_factor = sizes
            else:
                self.size_factor = dict.fromkeys(sizes, 1/np.sqrt(self.size))
        else:
            pass

    def get_item_array(self, users):
        return np.array([self.dic[user] for user in users])

    def get_size_factors(self, users):
        return np.array([self.size_factor[user] for user in users])


class BatchGenerator(object):
    def __init__(self, df, batch_size, users, items, ratings):
        self.batch_size = batch_size
        self.users = np.array(df[users])
        self.items = np.array(df[items])
        self.ratings = np.array(df[ratings])
        self.num_cols = len(df.columns)
        self.size = len(df)

    def get_batch(self):
        if self.size == 0:
            return np.array([]), np.array([]), np.array([])
        random_indices = np.random.randint(0, self.size, self.batch_size)
        users = self.users[random_indices]
        items = self.items[random_indices]
        ratings = self.ratings[random_indices]
        return users, items, ratings
