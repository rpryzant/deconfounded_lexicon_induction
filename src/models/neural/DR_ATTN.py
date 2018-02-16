"""
TODO -- REFACTOR THE SHIT OUT OF THIS!!!
"""

import numpy as np
import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
from tensorflow.python.framework import function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import src.msc.utils as utils

from src.models.abstract_model import Model
import tf_utils

class DR_ATTN:
    """ DR+ATTN """

    @staticmethod
    def build_model_graph(config, params, dataset):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(params)
            model = DR_ATTN(config, params, dataset, iterators)


        return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators):
        has_confounds = any(
            [(var['control'] and not var['skip']) \
            for var in config.data_spec[1:]])

        self.iter = iterators
        self.config = config
        self.params = params
        self.dataset = dataset

        self.learning_rate = tf.constant(params['learning_rate'])
        self.global_step = tf.Variable(0, trainable=False)
        self.dropout = tf.placeholder(tf.float32, name='dropout')

        source_name = dataset.input_varname()
        self.input_text, self.input_ids, self.input_lens = self.iter[source_name]

        # use attention to encode the input
        self.attn_scores, attn_context = self.attentional_encoder()
        # now get all the confounds into one vector
        confound_vector = self.vectorize_confounds()

        # use confounds to predict targets
        self.confound_output = defaultdict(dict)
        self.final_output = defaultdict(dict)
        for var in self.config.data_spec[1:]:
            if var['skip'] or var['control']:
                continue
            with tf.variable_scope(var['name']):
                if var['type'] == 'continuous':
                    confound_preds, confound_loss, final_preds, final_loss = \
                        self.double_predict_regression(
                            response=var,
                            confound_input=confound_vector,
                            x_input=attn_context)
                else:
                    confound_preds, confound_loss, final_preds, final_loss = \
                        self.double_predict_classification(
                            response=var,
                            confound_input=confound_vector,
                            x_input=attn_context)

            tf.summary.scalar('%s_confound_loss' % var['name'], confound_loss)
            self.confound_output[var['name']]['pred'] = confound_preds
            self.confound_output[var['name']]['loss'] = confound_loss

            tf.summary.scalar('%s_final_loss' % var['name'], final_loss)
            self.final_output[var['name']]['pred'] = final_preds
            self.final_output[var['name']]['loss'] = final_loss

        # add all yer losses up
        self.cum_confound_loss = tf.reduce_sum(
            [x['loss'] for x in self.confound_output.values()])
        self.cum_final_loss = tf.reduce_sum(
            [x['loss'] for x in self.final_output.values()])
        self.cumulative_loss = tf.reduce_sum(
            [self.cum_confound_loss, self.cum_final_loss])
        tf.summary.scalar('cum_confound_loss', self.cum_confound_loss)
        tf.summary.scalar('cum_final_loss', self.cum_final_loss)
        tf.summary.scalar('cum_loss', self.cumulative_loss)

        self.train_step = tf.contrib.layers.optimize_loss(
            loss=self.cumulative_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer='Adam',
            summaries=["loss", "gradient_norm"])

        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        self.trainable_variable_names = [v.name for v in tf.trainable_variables()]



    def attentional_encoder(self):
        # use attention to encode the source
        with tf.variable_scope('encoder'):
            rnn_outputs, source_embeddings = tf_utils.rnn_encode(
                source=self.input_ids,
                source_len=self.input_lens,
                vocab_size=self.dataset.vocab_size,
                embedding_size=self.params['embedding_size'],
                layers=self.params['encoder_layers'],
                units=self.params['encoder_units'],
                dropout=self.dropout,
                glove_matrix=tf_utils.get_glove(self.dataset) if self.params['use_glove'] else None)
        with tf.variable_scope('attention'):
            attn_scores, attn_context = tf_utils.attention(
                states=rnn_outputs,
                seq_lens=self.input_lens,
                layers=self.params['attn_layers'],
                units=self.params['attn_units'],
                dropout=self.dropout)

        return attn_scores, attn_context

    def vectorize_confounds(self):
        confounds = []
        for var in self.config.data_spec[1:]:
            if var['skip'] or not var['control']:
                continue
            if var['type'] == 'continuous':
                vals_as_cols = tf.expand_dims(self.iter[var['name']], 1)
                confounds.append(vals_as_cols)

            else:
                E = tf.get_variable(
                    name='%s_embeddings' % var['name'],
                    shape=[
                        self.dataset.num_levels(var['name']), 
                        self.params['embedding_size']])
                confounds.append(tf.nn.embedding_lookup(E, self.iter[var['name']]))
        confound_vector = tf.concat(confounds, axis=1)
        return confound_vector


    def double_predict_regression(self, response, confound_input, x_input):
        with tf.variable_scope('control_pred'):
            confound_preds, confound_loss = tf_utils.regressor(
                inputs=confound_input,
                labels=self.iter[response['name']],
                layers=self.params['regressor_layers'],
                hidden=self.params['regressor_units'],
                dropout=self.dropout)
        
        # force this into [batch size, embedding width + 1]
        final_input = tf.concat(
            [tf.expand_dims(confound_preds, 1), x_input], axis=1)

        if self.params['encoder_layers'] == 0:
            if self.params['use_glove']:
                x_input_dim = 50 # TODO set to a constant!!
            else:
                x_input_dim = self.params['embedding_size']
        else:
            x_input_dim = self.params['encoder_units'] * 2

        final_input = tf.reshape(final_input, [-1, x_input_dim + 1])

        with tf.variable_scope('final_pred'):
            final_preds, final_loss = tf_utils.regressor(
                inputs=final_input,
                labels=self.iter[response['name']],
                layers=self.params['regressor_layers'],
                hidden=self.params['regressor_units'],
                dropout=self.dropout)

        return confound_preds, confound_loss, final_preds, final_loss


    def double_predict_classification(self, response, confound_input, x_input):
        with tf.variable_scope('control_pred'):
            confound_preds, confound_loss = tf_utils.classifier(
                inputs=confound_input,
                labels=self.iter[response['name']],
                layers=self.params['classifier_layers'],
                num_classes=self.dataset.num_levels(response['name']),
                hidden=self.params['classifier_units'],
                dropout=self.dropout,
                sparse_labels=True)

        final_input = tf.concat([confound_preds, x_input], axis=1)

        with tf.variable_scope('final_pred'):
            final_preds, final_loss = tf_utils.classifier(
                inputs=final_input,
                labels=self.iter[response['name']],
                layers=self.params['classifier_layers'],
                num_classes=self.dataset.num_levels(response['name']),
                hidden=self.params['classifier_units'],
                dropout=self.dropout,
                sparse_labels=True)

        return confound_preds, confound_loss, final_preds, final_loss

    def train(self, sess):
        ops = [
            self.global_step,
            self.train_step,
            self.summaries
        ]
        return sess.run(ops, feed_dict={self.dropout: self.params['dropout']})


    def test(self, sess):
        # TODO  -- refactor, mostly shared with tf-flipper
        ops = [
            self.input_text,
            self.final_output,
            self.attn_scores
        ]
        inputs, outputs, scores = sess.run(ops, feed_dict={self.dropout: 0.0})

        attn_scores = defaultdict(list)
        for text, attn in zip(inputs, scores):
            for word, score in zip(text, attn):
                if word not in self.dataset.features:
                    continue
                attn_scores[word].append(score)

        if self.params['attn_importance_strategy'] == 'mean':
            attn_scores = {k: np.mean(v) for k, v in attn_scores.items()}
        elif self.params['attn_importance_strategy'] == 'max':
            attn_scores = {k: np.max(v) for k, v in attn_scores.items()}
        elif self.params['attn_importance_strategy'] == 'standardized_mean':
            attn_scores = {k: utils.standardize(v) for k, v in attn_scores.items()}
            attn_scores = {k: np.max(v) for k, v in attn_scores.items()}
        elif self.params['attn_importance_strategy'] == 'standardized_max':
            attn_scores = {k: utils.standardize(v) for k, v in attn_scores.items()}
            attn_scores = {k: np.max(v) for k, v in attn_scores.items()}

        output_scores = {
            response: output['pred'] \
            for response, output in outputs.items()}

        return output_scores, attn_scores
