import os
import sys
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import functools
# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

# STD_EPSILON = 1e-07

class ReiteratableWrapper(object):
    def __init__(self, f):
        self.f = f
    def __iter__(self):
        return self._f()


class VAE(object):
    """docstring for VAE"""
    def __init__(self, input_dim=784, hidden_dim=300, z_dim=2):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.z_dim = z_dim
        self.batch_size = 64

        self.x = tf.placeholder(tf.float32, [None, self.input_dim])

        # encoder
        self.encoder = tf.layers.dense(self.x, units=self.hidden_dim)
        self.encoder = tf.layers.batch_normalization(self.encoder, training=True)
        self.encoder = tf.nn.softplus(self.encoder)
        self.encoder = tf.layers.dense(self.encoder, units=self.hidden_dim)
        self.encoder = tf.layers.batch_normalization(self.encoder, training=True)

        self.encoder_mean = tf.nn.softplus(self.encoder)
        self.encoder_mean = tf.layers.dense(self.encoder_mean, units=self.hidden_dim)
        self.encoder_mean = tf.layers.batch_normalization(self.encoder_mean, training=True)
        self.encoder_mean = tf.nn.softplus(self.encoder_mean)
        self.encoder_mean = tf.layers.dense(self.encoder_mean, units=self.z_dim)

        self.encoder_std = tf.layers.dense(self.encoder, units=self.hidden_dim)
        self.encoder_std = tf.layers.batch_normalization(self.encoder_std, training=True)
        self.encoder_std = tf.nn.softplus(self.encoder_std)
        self.encoder_std = tf.layers.dense(self.encoder_std, units=self.z_dim, activation=tf.nn.softplus)

        # sampling
        self.epsilon = tf.random_normal(shape=(self.z_dim,), mean=0.0, stddev=1.0)
        self.z = self.encoder_mean + tf.exp(self.encoder_std / 2.0) * self.epsilon

        # decoder
        self.decoder = tf.layers.dense(self.z, units=self.hidden_dim, name='fc1')
        self.decoder = tf.layers.batch_normalization(self.decoder, training=True, name='bn1')
        self.decoder = tf.nn.softplus(self.decoder)
        self.decoder = tf.layers.dense(self.decoder, units=self.hidden_dim, name='fc2')
        self.decoder = tf.layers.batch_normalization(self.decoder, training=True, name='bn2')
        self.decoder = tf.layers.dense(self.decoder, units=self.input_dim, activation=tf.sigmoid, name='output')

    # ここが違いそう
    def loss_func(self, x, x_hat):
        kl_div = 1 + self.encoder_std - tf.square(self.encoder_mean) - tf.exp(self.encoder_std)
        kl_div = -0.5 * tf.reduce_sum(kl_div, 1)

        log_likehood = x * tf.log(1e-10 + x_hat) + (1 - x) * tf.log(1e-10 + 1 - x_hat)
        log_likehood = -tf.reduce_sum(log_likehood, 1)

        tf.summary.tensor_summary('loss', kl_div + log_likehood)

        return tf.reduce_mean(kl_div + log_likehood)

    def vae_train(self, batch_size, steps, lr, mn, save_dir):
        self.batch_size = batch_size

        loss_op = self.loss_func(self.x, self.decoder)
        train_op = tf.train.MomentumOptimizer(learning_rate=lr, momentum=mn).minimize(loss_op)

        ''' generatorを引数にしたい
        f = functools.partial(x_data_generator, batch_size)
        reiteratable_x_generator = ReiteratableWrapper(f)
        '''

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter('vae_log', graph=sess.graph)

            saver = tf.train.Saver()

            ''' generatorを引数にしたい
            for i in range(epoch):
                for batch_x in reiteratable_x_generator:
                    feed_dict = {self.x: batch_x, self.z:sampling(self.encoder_mean, self.encoder_std)}
                    _, l, result = sess.run([train_op, loss_op, summary_op], feed_dict=feed_dict)
            '''

            # minist sample
            mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
            for i in range(steps):
                batch_x, _ = mnist.train.next_batch(batch_size)
                feed_dict = {self.x: batch_x}
                _, l, result = sess.run([train_op, loss_op, summary_op], feed_dict=feed_dict)

                summary_writer.add_summary(result, i)

                if i % 1000 == 0:
                    print(f'steps {i}, Loss: {l}')

                    '''
                    save_path = os.path.join(save_dir + f'model_epoch:{i}_loss:{l}.ckpt')
                    saver.save(sess, save_path)
                    '''

            save_path = os.path.join(save_dir + "figure.png")
            generate_num = 20

            x_axis = np.linspace(-30000, 30000, generate_num)
            y_axis = np.linspace(-30000, 30000, generate_num)

            canvas = np.empty((28 * generate_num, 28 * generate_num))

            for i, yi in enumerate(x_axis):
                for j, xi in enumerate(y_axis):
                    z_mu = np.array([[xi, yi]] * self.batch_size)
                    x_mean = sess.run(self.decoder, feed_dict={self.z: z_mu})
                    canvas[(generate_num - i - 1) * 28:(generate_num - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

            plt.figure(figsize=(8, 10))
            Xi, Yi = np.meshgrid(x_axis, y_axis)
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.savefig(save_path)

'''
    def generate_image(self, generate_num, save_dir, batch_size):
        self.batch_size = batch_size
        self.generate_input = tf.placeholder(tf.float32, [None, self.z_dim])

        # trained_decoder
        self.trained_decoder = tf.layers.dense(self.generate_input, units=self.hidden_dim, name='fc1', reuse=True)
        self.trained_decoder = tf.layers.batch_normalization(self.trained_decoder, trainable=True, name='bn1', reuse=True)
        self.trained_decoder = tf.nn.softplus(self.trained_decoder)
        self.trained_decoder = tf.layers.dense(self.trained_decoder, units=self.hidden_dim, name='fc2', reuse=True)
        self.trained_decoder = tf.layers.batch_normalization(self.trained_decoder, trainable=True, name='bn2', reuse=True)
        self.trained_decoder = tf.layers.dense(self.trained_decoder, units=self.input_dim, activation=tf.sigmoid, name='output', reuse=True)

        save_path = os.path.join(save_dir + "figure.png")

        x_axis = np.linspace(-1, 1, generate_num)
        y_axis = np.linspace(-1, 1, generate_num)

        canvas = np.empty((28 * generate_num, 28 * generate_num))

        init = tf.initialize_all_variables()
        with tf.Session() as sess:
            sess.run(init)

            for i, yi in enumerate(x_axis):
                for j, xi in enumerate(y_axis):
                    z_mu = np.array([[xi, yi]] * self.batch_size)
                    x_mean = sess.run(self.trained_decoder, feed_dict={self.generate_input: z_mu})
                    canvas[(generate_num - i - 1) * 28:(generate_num - i) * 28, j * 28:(j + 1) * 28] = x_mean[0].reshape(28, 28)

            plt.figure(figsize=(8, 10))
            Xi, Yi = np.meshgrid(x_axis, y_axis)
            plt.imshow(canvas, origin="upper", cmap="gray")
            plt.savefig(save_path)
'''