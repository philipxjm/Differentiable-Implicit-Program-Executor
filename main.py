import model
import tensorflow as tf
from data import read_sdf_dir, read_program_dir, generate_data
import numpy as np
import sys
from tqdm import tqdm
import hyper_params as hp

# np.set_printoptions(threshold=sys.maxsize)


def train(model, dataset, epochs, save_name):
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    pbar = tqdm(range(0, epochs))
    programs = dataset["programs"]
    positions = dataset["positions"]
    labels = dataset["labels"]
    print(len(programs))
    # for i in pbar:
    #     l, _ = sess.run([model.loss, model.optimize],
    #                     feed_dict={model.inputs: x,
    #                                model.labels: y,
    #                                model.dropout: hp.KEEP_PROB})
    #     pbar.set_description("epoch {}, loss={}".format(i, l))
    #     if i % 100 == 0:
    #         print("epoch {}, loss={}".format(i, l))
    #     if i % 500 == 0:
    #         print("Saving at epoch {}, loss={}".format(i, l))
    #         saver.save(sess,
    #                    save_name + str(l),
    #                    global_step=i)
    #     if i % 1000 == 0:
    #         total_correct = 0
    #         total_symbols = 0
    #         for piece in pieces["test"]:
    #             x = np.expand_dims(piece[:-1], axis=0)
    #             y = np.expand_dims(piece[1:], axis=0)
    #             prediction = sess.run(model.logits,
    #                                   feed_dict={model.inputs: x,
    #                                              model.dropout: 1.0})
    #             activation = np.argmax(prediction, axis=2)
    #             # print("act: ", activation)
    #             # print("lab: ", y)
    #             total_correct += np.sum(y == activation)
    #             total_symbols += activation.shape[1]
    #         print(total_correct / total_symbols)
    # final_loss = sess.run([model.loss],
    #                       feed_dict={model.inputs: x,
    #                                  model.labels: y})
    # saver.save(sess, save_name + str(final_loss[0]))


if __name__ == '__main__':
    programs = tf.placeholder(tf.int32, shape=[None, None])
    positions = tf.placeholder(tf.float32, shape=[None, 3])
    labels = tf.placeholder(tf.int32, shape=[None, 1])
    dropout = tf.placeholder(tf.float32, shape=())

    padded_sign_matrices = read_sdf_dir("data/sdf_small/")
    programs = read_program_dir("data/programs/")
    data_iter = generate_data(padded_sign_matrices, programs)
    for i in data_iter:
        print(i[0])
    m = model.Model(programs=programs,
                    positions=positions,
                    labels=labels,
                    dropout=dropout)
    # train(m, training_set, 10, "model/jsb8/model_")
    # test(m, pieces, "model/jsb8/model")
    # generate(m, pieces, "model/jsb8/model", token2idx, idx2token)
