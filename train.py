#! /usr/bin/env python

import tensorflow as tf
import numpy as np
import os
import time
import datetime
import re
import data_helpers
from gensim.models import FastText
from text_cnn import TextCNN
from tensorflow.contrib import learn

# Parameters
# ==================================================

flags = tf.app.flags
FLAGS = flags.FLAGS

# Data loading parameters
flags.DEFINE_float("dev_sample_percentage", .1, "Percentage of the training data to use for validation")
flags.DEFINE_string("positive_data_file", "./data/rt-polaritydata/rt-polarity.pos", "Data source for the positive data")
flags.DEFINE_string("negative_data_file", "./data/rt-polaritydata/rt-polarity.neg", "Data source for the negative data")
flags.DEFINE_string("character_encoding", "utf-8", "Input file encoding.")

# Pretrained embeddings paramameters
flags.DEFINE_string("word2vec", None, "Word2vec file with pre-trained embeddings")
flags.DEFINE_string("fasttext", None, "fastText file with pre-trained embeddings")
flags.DEFINE_string("conceptnet_numberbatch", None, "conceptnet-numberbatch file with pre-trained embeddings")
flags.DEFINE_string("conceptnet_numberbatch_lang_filter", "\w*", "conceptnet-numberbatch language filter regular expression")

# Model hyperparameters
flags.DEFINE_integer("embedding_dim", 128, "Dimensionality of character embedding")
flags.DEFINE_string("filter_sizes", "3,4,5", "Comma-separated filter sizes")
flags.DEFINE_integer("num_filters", 128, "Number of filters per filter size")
flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularization lambda")

# Training parameters
flags.DEFINE_integer("batch_size", 64, "Batch Size")
flags.DEFINE_integer("num_epochs", 25, "Number of training epochs")
flags.DEFINE_integer("evaluate_every", 100, "Evaluate model on dev set after this many steps")

# Misc parameters
flags.DEFINE_boolean("allow_soft_placement", True, "Allow soft device placement")
flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
flags.DEFINE_string("logs_subfolder", "", "Subfolder within 'runs'-folder")


def main(unused_argv):
    # print("\nParameters:")
    # for k, v in FLAGS.flag_values_dict().items():
    #     print("{}={}".format(k.upper(), v))
    # print("")

    # Data Preparation
    # ==================================================

    # Load data
    print("Loading data...")
    x_text, y = data_helpers.load_data_and_labels(FLAGS.positive_data_file
        , FLAGS.negative_data_file, FLAGS.character_encoding)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))

    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(FLAGS.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]

    del x, y, x_shuffled, y_shuffled

    vocabulary_size = len(vocab_processor.vocabulary_)
    print("Vocabulary Size: {:d}".format(vocabulary_size))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))


    # Training
    # ==================================================

    with tf.Graph().as_default():
        session_conf = tf.ConfigProto(
          allow_soft_placement=FLAGS.allow_soft_placement,
          log_device_placement=FLAGS.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            cnn = TextCNN(
                sequence_length=x_train.shape[1],
                num_classes=y_train.shape[1],
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=FLAGS.embedding_dim,
                filter_sizes=list(map(int, FLAGS.filter_sizes.split(","))),
                num_filters=FLAGS.num_filters,
                l2_reg_lambda=FLAGS.l2_reg_lambda)

            # Define Training procedure
            global_step = tf.Variable(0, name="global_step", trainable=False)
            optimizer = tf.train.AdamOptimizer(1e-3)
            grads_and_vars = optimizer.compute_gradients(cnn.loss)
            train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

            # Keep track of gradient values and sparsity (optional)
            grad_summaries = []
            for g, v in grads_and_vars:
                if g is not None:
                    grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                    sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                    grad_summaries.append(grad_hist_summary)
                    grad_summaries.append(sparsity_summary)
            grad_summaries_merged = tf.summary.merge(grad_summaries)

            # Output directory for models and summaries
            timestamp = str(int(time.time()))
            out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", FLAGS.logs_subfolder, timestamp))
            print("Writing to {}\n".format(out_dir))

            # Summaries for loss and accuracy
            loss_summary = tf.summary.scalar("loss", cnn.loss)
            acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

            # Train Summaries
            train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
            train_summary_dir = os.path.join(out_dir, "summaries", "train")
            train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

            # Dev summaries
            dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
            dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
            dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

            # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
            checkpoint_dir = os.path.abspath(os.path.join(out_dir, "best-model"))
            checkpoint_prefix = os.path.join(checkpoint_dir, "model")
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

            # Write vocabulary
            vocab_processor.save(os.path.join(out_dir, "vocab"))

            # Initialize all variables
            sess.run(tf.global_variables_initializer())

            if FLAGS.word2vec:
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
                print("Loading word2vec embeddings file {}\n".format(FLAGS.word2vec))
                with open(FLAGS.word2vec, "rb") as f:
                    header = f.readline()
                    emb_vocab_size, layer1_size = map(int, header.split())
                    assert layer1_size == FLAGS.embedding_dim
                    binary_len = np.dtype('float32').itemsize * layer1_size
                    coverage = 0
                    for line in range(emb_vocab_size):
                        word = []
                        while True:
                            ch = f.read(1)
                            if ch == b' ':
                                break
                            if ch != b'\n':
                                word.append(ch)
                        word = str(b''.join(word), encoding='utf-8', errors='strict')
                        idx = vocab_processor.vocabulary_.get(word)
                        if idx != 0:
                            initW[idx] = np.fromstring(f.read(binary_len), dtype='float32')
                            coverage += 1
                        else:
                            f.seek(binary_len, 1)
                        if(line % 10000 == 0 or line == (emb_vocab_size-1)):
                            print('Embeddings scan progress: {0:.2%}, coverage: {1:.2%}'\
                                .format(1. * line / emb_vocab_size, 1. * coverage / vocabulary_size))
                sess.run(cnn.W.assign(initW))

            if FLAGS.fasttext:
                initW = np.random.uniform(-0.25, 0.25, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
                print("Loading fasttext embeddings file {}\n".format(FLAGS.fasttext))
                ftm = FastText.load_fasttext_format(model_file=FLAGS.fasttext, encoding="utf-8")
                assert FLAGS.embedding_dim == ftm.wv.vector_size
                vocab_dict = vocab_processor.vocabulary_._mapping
                sorted_vocab = sorted(vocab_dict.items(), key = lambda x : x[1])
                coverage = 0
                t_cnt = 0
                for t, idx in sorted_vocab:
                    t_cnt += 1
                    if t.lower() in ftm.wv.vocab and t != u'<UNK>':
                        v = ftm[t.lower()]
                        assert v.dtype == 'float32'
                        initW[idx] = v
                        coverage += 1
                    if(t_cnt % 10000 == 0 or t_cnt == (vocabulary_size - 1)):
                        print('Embeddings file scan progress: {0:.2%}, coverage: {1:.2%}'\
                            .format(1. * t_cnt / vocabulary_size, 1. * coverage / vocabulary_size))
                sess.run(cnn.W.assign(initW))

            if FLAGS.conceptnet_numberbatch:
                initW = np.random.normal(0, 0.06, (len(vocab_processor.vocabulary_), FLAGS.embedding_dim))
                print("Loading conceptnet_numberbatch embeddings file {}\n".format(FLAGS.conceptnet_numberbatch))
                with open(FLAGS.conceptnet_numberbatch, "r", encoding=FLAGS.character_encoding) as f:
                    header = f.readline()
                    emb_vocab_size, layer1_size = map(int, header.split())
                    assert  FLAGS.embedding_dim == layer1_size
                    coverage = 0
                    lang_stats = dict()
                    for i in range(emb_vocab_size):
                        line = f.readline()
                        word_search = re.search(r'/c/('+ FLAGS.conceptnet_numberbatch_lang_filter + ')/(\w*)\s(.*)', line)
                        if word_search:
                            lang = word_search.group(1)
                            word = word_search.group(2)
                            vector = np.fromstring(word_search.group(3), dtype='float32', sep=u' ')
                            idx = vocab_processor.vocabulary_.get(word)
                            if idx != 0:
                                initW[idx] = vector
                                coverage += 1
                                # print("lang: {}, word: {}".format(lang, word))
                                if lang in lang_stats:
                                    lang_stats[lang] += 1
                                else:
                                    lang_stats[lang] = 1
                            if(i % 10000 == 0 or i == (emb_vocab_size-1)):
                                print('Embeddings scan progress: {0:.2%}, coverage: {1:.2%}'\
                                    .format(1. * i / emb_vocab_size, 1. * coverage / vocabulary_size))
                    print(lang_stats)
                sess.run(cnn.W.assign(initW))

            def train_step(x_batch, y_batch):
                """
                A single training step
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
                }
                _, step, summaries, loss, accuracy = sess.run(
                    [train_op, global_step, train_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                train_summary_writer.add_summary(summaries, step)

            def dev_step(x_batch, y_batch, writer=None):
                """
                Evaluates model on a dev set
                """
                feed_dict = {
                  cnn.input_x: x_batch,
                  cnn.input_y: y_batch,
                  cnn.dropout_keep_prob: 1.0
                }
                step, summaries, loss, accuracy = sess.run(
                    [global_step, dev_summary_op, cnn.loss, cnn.accuracy],
                    feed_dict)
                time_str = datetime.datetime.now().isoformat()
                print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
                if writer:
                    writer.add_summary(summaries, step)
                return accuracy

            # Generate batches
            batches = data_helpers.batch_iter(
                list(zip(x_train, y_train)), FLAGS.batch_size, FLAGS.num_epochs)
            # Training loop. For each batch...
            best_perf = -1.0 # larger is better
            for batch in batches:
                x_batch, y_batch = zip(*batch)
                train_step(x_batch, y_batch)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    acc = dev_step(x_dev, y_dev, writer=dev_summary_writer)
                    print("")
                    if acc > best_perf:
                        best_perf = acc
                        path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                        print("Saved model to {}\n".format(path))


if __name__ == '__main__':
    tf.app.run()
