import numpy as np
from datetime import datetime, timedelta
from py2neo import Graph, Relationship, Node


def present_status(steps, duration):
    seconds = timedelta(seconds=int(duration))
    date_time = datetime(1, 1, 1) + seconds
    print('Duration of whole training with % s steps is %.2f seconds,'
          % (steps, duration))
    print(" (DAYS:HOURS:MIN:SEC)")


def mse(x, y):
    assert len(x) == len(y)
    return np.mean(np.power(x - y, 2))


def rmse(x, y):
    return np.sqrt(mse(x, y))


def put_data_frame_in_db(df):
    graph = Graph(password="cyclops")
    for row in df.itertuples():
        user = Node('User', id=row.user.item())
        deal = Node('Deal', id=row.deal.item())
        graph.merge(user)
        graph.merge(deal)
        user.push()
        graph.create(Relationship(user, "rates", deal, rating=row.rating))


def get_deals_not_rated_by_this_customer(df, users, deals, user):
    graph = Graph(password="cyclops")
    query = """
    MATCH (u:User) - [:rates] ->  (d:Deal)
    WHERE u.id = {user_id}
    RETURN d
    """
    user_deals = graph.run(query, user_id=user)
    deals_rated = np.array([deal['d']['id'] for deal in user_deals])
    all_deals = np.array(df[deals].unique())
    return np.setdiff1d(all_deals, deals_rated)


def get_np_redundant_array(size, user):
    return [user] * size
