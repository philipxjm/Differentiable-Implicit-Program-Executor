import tensorflow as tf
import hyper_params as hp
from modules import linear, lrelu


class Model:
    def __init__(self, programs, positions, labels, dropout):
        self.programs = programs
        self.positions = positions
        self.labels = labels
        self.dropout = dropout

        self.encoding = self.encoder(programs)[0][:, -1, :]
        self.logits = self.decoder(self.encoding, self.positions)
        self.optimize, self.loss = self.train(self.logits, self.labels)

    def encoder(self,
                programs,
                rnn_size=hp.RNN_SIZE,
                num_layers=hp.RNN_LAYER_COUNT):
        embed = tf.contrib.layers.embed_sequence(programs,
                                                 vocab_size=hp.VOCAB_SIZE,
                                                 embed_dim=hp.EMBEDDING_SIZE)
        stacked_cells = tf.contrib.rnn.MultiRNNCell(
            [tf.contrib.rnn.DropoutWrapper(tf.contrib.rnn.LSTMCell(rnn_size),
                                           self.dropout)
             for _ in range(num_layers)]
        )
        outputs, state = tf.nn.dynamic_rnn(stacked_cells,
                                           embed,
                                           dtype=tf.float32)
        return outputs, state

    def decoder(self,
                encoding,
                positions,
                hidden_size=hp.HIDDEN_SIZE):
        with tf.variable_scope("simple_net"):
            pointz = tf.concat([positions, encoding], 1)
            print("pointz", pointz.shape)

            h1 = lrelu(linear(pointz, hidden_size*16, 'h1_lin'))
            h1 = tf.concat([h1, pointz], 1)

            h2 = lrelu(linear(h1, hidden_size*8, 'h4_lin'))
            h2 = tf.concat([h2, pointz], 1)

            h3 = lrelu(linear(h2, hidden_size*4, 'h5_lin'))
            h3 = tf.concat([h3, pointz], 1)

            h4 = lrelu(linear(h3, hidden_size*2, 'h6_lin'))
            h4 = tf.concat([h4, pointz], 1)

            h5 = lrelu(linear(h4, hidden_size, 'h7_lin'))
            h6 = tf.nn.sigmoid(linear(h5, 1, 'h8_lin'))

            return tf.reshape(h6, [-1, 1])

    def loss_function(self, logits, labels):
        loss = tf.reduce_mean(tf.square(logits - labels))
        return loss

    def train(self, logits, labels):
        loss = self.loss_function(logits, labels)
        return tf.train.AdamOptimizer(1e-4).minimize(loss), loss
