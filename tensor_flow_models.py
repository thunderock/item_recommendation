import numpy as np
import tensorflow as tf
from utilities import present_status, rmse
import time
import os


def svd_inference(user_batch, item_batch, user_num, item_num, dim=5):
    with tf.name_scope('Declaring_variables'):
        global_bias = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        w_user = tf.get_variable("embd_user", shape=[user_num, dim], initializer=initializer)
        w_item = tf.get_variable("embd_item", shape=[item_num, dim], initializer=initializer)
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embedding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embedding_item")

    with tf.name_scope('Prediction_regularizer'):
        infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
        infer = tf.add(infer, global_bias)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        l2_user = tf.sqrt(tf.nn.l2_loss(embd_user))
        l2_item = tf.sqrt(tf.nn.l2_loss(embd_item))
        bias_user_sq = tf.square(bias_user)
        bias_item_sq = tf.square(bias_item)
        bias_sum = tf.add(bias_user_sq, bias_item_sq)
        l2_sum = tf.add(l2_user, l2_item)
        regularizer = tf.add(l2_sum, bias_sum, name="svd_regularizer")
    dic_of_values = {'infer': infer,
                     'regularizer': regularizer,
                     'w_user': w_user,
                     'w_item': w_item}
    return dic_of_values


def nsvd_inference(user_batch,
                   item_batch,
                   user_item_batch,
                   size_factor,
                   user_num,
                   item_num,
                   dim=5):
    with tf.name_scope('Declaring_variables'):
        bias_global = tf.get_variable("bias_global", shape=[])
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        bias_user = tf.nn.embedding_lookup(w_bias_user,
                                           user_batch,
                                           name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item,
                                           item_batch,
                                           name="bias_item")
        initializer = tf.truncated_normal_initializer(stddev=0.02)
        w_item1 = tf.get_variable(name='w_item1',
                                  shape=[item_num, dim],
                                  initializer=initializer)
        w_item2 = tf.get_variable(name='w_item2',
                                  shape=[item_num, dim],
                                  initializer=initializer)
        zero = np.zeros(dim)
        fake_item = tf.constant(np.array([zero]), dtype=tf.float32, shape=[1,dim], name='fake')
        w_item2 = tf.concat([w_item2, fake_item], 0)
        embd_item1 = tf.nn.embedding_lookup(w_item1, item_batch)
        embd_item2 = tf.nn.embedding_lookup(w_item2, user_item_batch)
        embd_item2 = tf.transpose(tf.reduce_sum(embd_item2, 1))
        embd_item2 = tf.transpose(tf.multiply(embd_item2, size_factor))
    with tf.name_scope('Prediction_regularizer'):
        infer = tf.reduce_sum(tf.multiply(embd_item1, embd_item2), 1)
        infer = tf.add(infer, bias_global)
        infer = tf.add(infer, bias_user)
        infer = tf.add(infer, bias_item, name="svd_inference")
        l2_user = tf.sqrt(tf.nn.l2_loss(embd_item1))
        l2_item = tf.sqrt(tf.nn.l2_loss(embd_item2))
        bias_user_sq = tf.square(bias_user)
        bias_item_sq = tf.square(bias_item)
        bias_sum = tf.add(bias_user_sq, bias_item_sq)
        l2_sum = tf.add(l2_user, l2_item)
        regularizer = tf.add(l2_sum, bias_sum, name="svd_regularizer")
    dic_of_values = {'infer': infer,
                     'regularizer': regularizer,
                     'w_item1': w_item1,
                     'w_item2': w_item2}
    return dic_of_values


def loss_function(infer, regularizer, rate_batch, reg):
    cost_l2 = tf.square(tf.subtract(rate_batch, infer))
    lambda3 = tf.constant(reg, dtype=tf.float32, shape=[], name="lambda3")
    cost = tf.add(cost_l2, tf.multiply(regularizer, lambda3))
    return cost


class SVDTrainer(object):
    def __init__(self,
                 num_of_users,
                 num_of_items,
                 train_batch_generator,
                 test_batch_generator,
                 valid_batch_generator,
                 finder=None,
                 model="svd"):
        self.num_of_users = num_of_users
        self.num_of_items = num_of_items
        self.train_batch_generator = train_batch_generator
        self.test_batch_generator = test_batch_generator
        self.valid_batch_generator = valid_batch_generator
        self.valid_batch_generator = valid_batch_generator
        self.model = model
        self.finder = finder
        self.general_duration = 0
        self.num_steps = 0
        self.dimension = None
        self.regularizer = None
        self.best_acc_test = float('inf')

    def set_graph(self,
                  hp_dim,
                  hp_reg,
                  learning_rate,
                  momentum_factor):
        self.dimension = hp_dim
        self.regularizer = hp_reg
        self.learning_rate = learning_rate
        self.momentum_factor = momentum_factor
        self.graph = tf.Graph()
        with self.graph.as_default():

            # Placeholders
            self.tf_user_batch = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="id_user")
            self.tf_item_batch = tf.placeholder(tf.int32,
                                                shape=[None],
                                                name="id_item")
            self.tf_rate_batch = tf.placeholder(tf.float32,
                                                shape=[None],
                                                name="actual_ratings")
            if self.model == "nsvd":
                self.tf_user_item = tf.placeholder(tf.int32,
                                                   shape=[None, None],
                                                   name="user_item")
                self.tf_size_factor = tf.placeholder(tf.float32,
                                                     shape=[None],
                                                     name="size_factor")

            if self.model == "nsvd":
                tf_svd_model = nsvd_inference(self.tf_user_batch,
                                              self.tf_item_batch,
                                              self.tf_user_item,
                                              self.tf_size_factor,
                                              user_num=self.num_of_users,
                                              item_num=self.num_of_items,
                                              dim=hp_dim)
            else:
                tf_svd_model = svd_inference(self.tf_user_batch,
                                             self.tf_item_batch,
                                             user_num=self.num_of_users,
                                             item_num=self.num_of_items,
                                             dim=hp_dim)
            self.infer = tf_svd_model['infer']
            regularizer = tf_svd_model['regularizer']
            global_step = tf.contrib.framework.get_or_create_global_step()

            with tf.name_scope('loss'):
                self.tf_cost = loss_function(self.infer,
                                             regularizer,
                                             self.tf_rate_batch,
                                             reg=hp_reg)

            # Optimizer
            with tf.name_scope('training'):
                global_step = tf.contrib.framework.assert_or_get_global_step()
                assert global_step is not None
                optimizer = tf.train.MomentumOptimizer(learning_rate,
                                                       momentum_factor)
                self.train_op = optimizer.minimize(self.tf_cost,
                                                   global_step=global_step)

            # Saver
            self.saver = tf.train.Saver()
            save_dir = 'checkpoints/'
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            self.save_path = os.path.join(save_dir, 'best_validation')

            # Minibatch accuracy using rmse
            with tf.name_scope('accuracy'):
                difference = tf.pow(tf.subtract(self.infer, self.tf_rate_batch), 2)
                self.acc_op = tf.sqrt(tf.reduce_mean(difference))

    def training(self,
                 hp_dim,
                 hp_reg,
                 learning_rate,
                 momentum_factor,
                 num_steps,
                 verbose=True):
        self.set_graph(hp_dim,
                       hp_reg,
                       learning_rate,
                       momentum_factor)

        self.num_steps = num_steps
        marker = ''

        with tf.Session(graph=self.graph) as sess:
            tf.initialize_all_variables().run()
            if verbose:
                print("{} {} {} {}".format("step",
                                           "batch_error",
                                           "test_error",
                                           "elapsed_time"))
            else:
                print("\nTraining")
            start = time.time()
            initial_time = start
            for step in range(num_steps):
                users, items, rates = self.train_batch_generator.get_batch()
                if self.model == "nsvd":
                    items_per_user = self.finder.get_item_array(users)
                    size_factor = self.finder.get_size_factors(users)
                    f_dict = {self.tf_user_batch: users,
                              self.tf_item_batch: items,
                              self.tf_rate_batch: rates,
                              self.tf_size_factor: size_factor,
                              self.tf_user_item: items_per_user}
                else:
                    f_dict = {self.tf_user_batch: users,
                              self.tf_item_batch: items,
                              self.tf_rate_batch: rates}
                _, pred_batch, cost, train_error = sess.run([self.train_op,
                                                             self.infer,
                                                             self.tf_cost,
                                                             self.acc_op],
                                                            feed_dict=f_dict)
                if (step % 1000) == 0:
                    users, items, rates = self.test_batch_generator.get_batch()
                    if self.model == "nsvd":
                        items_per_user = self.finder.get_item_array(users)
                        size_factor = self.finder.get_size_factors(users)
                        f_dict = {self.tf_user_batch: users,
                                  self.tf_item_batch: items,
                                  self.tf_rate_batch: rates,
                                  self.tf_size_factor: size_factor,
                                  self.tf_user_item: items_per_user}
                    else:
                        f_dict = {self.tf_user_batch: users,
                                  self.tf_item_batch: items,
                                  self.tf_rate_batch: rates}
                    pred_batch = sess.run(self.infer, feed_dict=f_dict)
                    test_error = rmse(pred_batch, rates)
                    if test_error < self.best_acc_test:
                        self.best_acc_test = test_error
                        marker = "*"
                        self.saver.save(sess=sess, save_path=self.save_path)

                    end = time.time()
                    if verbose:
                        print("{:3d} {:f} {:f}{:s} {:f}(s)".format(step,
                                                                   train_error,
                                                                   test_error,
                                                                   marker,
                                                                   end -
                                                                   start))
                    marker = ''
                    start = end
        self.general_duration = time.time() - initial_time

    def print_stats(self):
        present_status(self.num_steps, self.general_duration)

    def prediction(self,
                   users_list=None,
                   list_of_items=None,
                   show_valid=False):
        if self.dimension is None and self.regularizer is None:
            print("You can not have a prediction without training!!!!")
        else:
            self.set_graph(self.dimension,
                           self.regularizer,
                           self.learning_rate,
                           self.momentum_factor)
            with tf.Session(graph=self.graph) as sess:
                self.saver.restore(sess=sess, save_path=self.save_path)
                users, items, rates = self.valid_batch_generator.get_batch()
                if show_valid:
                    if self.model == "nsvd":
                        items_per_user = self.finder.get_item_array(users)
                        size_factor = self.finder.get_size_factors(users)
                        f_dict = {self.tf_user_batch: users,
                                  self.tf_item_batch: items,
                                  self.tf_rate_batch: rates,
                                  self.tf_size_factor: size_factor,
                                  self.tf_user_item: items_per_user}
                    else:
                        f_dict = {self.tf_user_batch: users,
                                  self.tf_item_batch: items,
                                  self.tf_rate_batch: rates}
                    valid_error = sess.run(self.acc_op, feed_dict=f_dict)
                    return valid_error
                else:
                    if self.model == "nsvd":
                        items_per_user = self.finder.get_item_array(users_list)
                        size_factor = self.finder.get_size_factors(users_list)
                        f_dict = {self.tf_user_batch: users_list,
                                  self.tf_item_batch: list_of_items,
                                  self.tf_size_factor: size_factor,
                                  self.tf_user_item: items_per_user}
                    else:
                        f_dict = {self.tf_user_batch: users_list,
                                  self.tf_item_batch: list_of_items}
                    prediction = sess.run(self.infer, feed_dict=f_dict)
                    sess.close()
                    return prediction
