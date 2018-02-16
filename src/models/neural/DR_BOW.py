
import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
from tensorflow.python.framework import function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model
import tf_utils

class DR_BOW:

    @staticmethod
    def build_model_graph(config, params, dataset):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(params)
            model = DR_BOW(config, params, dataset, iterators)

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


        # transform input text into big BOW vector
        with tf.variable_scope('input'):
            input_vector = tf.map_fn(
                lambda seq: self._to_dense_vector(seq, self.dataset.vocab_size),
                self.iter[dataset.input_varname()][1])
            input_encoded = tf_utils.fc_tube(
                inputs=tf.cast(input_vector, tf.float32),
                num_outputs=self.params['encoding_dim'],
                layers=self.params['encoder_layers'])
        # TODO this is PAINFULLY hacky!!!
        cur_graph = tf.get_default_graph()
        self.feature_weights = cur_graph.get_tensor_by_name(
            'input/layer_0/weights:0')
        self.feature_intercept = cur_graph.get_tensor_by_name(
            'input/layer_0/biases:0')

        # transform confounds similarly
        with tf.variable_scope('condfound_input'):
            confound_vector = self.vectorize_confounds()

        # now get all the confounds into one vector
        confound_vector = self.vectorize_confounds()

        # use confounds to predict targets
        # TODO -- LOTS OF SHARED CODE WITH TF_CAUSAL!!!
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
                            x_input=input_encoded)
                else:
                    confound_preds, confound_loss, final_preds, final_loss = \
                        self.double_predict_classification(
                            response=var,
                            confound_input=confound_vector,
                            x_input=input_encoded)

            tf.summary.scalar('%s_confound_loss' % var['name'], confound_loss)
            self.confound_output[var['name']]['pred'] = confound_preds
            self.confound_output[var['name']]['loss'] = confound_loss

            tf.summary.scalar('%s_final_loss' % var['name'], final_loss)
            self.final_output[var['name']]['pred'] = final_preds
            self.final_output[var['name']]['loss'] = final_loss

        # regularize if need be
        if self.params['lambda'] > 0:
            if self.params['regularizor'] == 'l2':
                reg = tf.contrib.layers.l2_regularizer(self.params['lambda'])
            else:
                reg = tf.contrib.layers.l1_regularizer(self.params['lambda'])
            reg_weights = tf.trainable_variables() \
                if self.params['reg_type'] =='all' else [self.feature_weights]
            reg_term = tf.contrib.layers.apply_regularization(reg, reg_weights)
        else:
            reg_term = 0

        # add all yer losses up
        self.cum_confound_loss = tf.reduce_sum(
            [x['loss'] for x in self.confound_output.values()])
        self.cum_final_loss = tf.reduce_sum(
            [x['loss'] for x in self.final_output.values()])
        self.cumulative_loss = tf.reduce_sum(
            [self.cum_confound_loss, self.cum_final_loss])
        self.cumulative_loss += reg_term

        tf.summary.scalar('regularization_loss', reg_term)
        tf.summary.scalar('cum_confound_loss', self.cum_confound_loss)
        tf.summary.scalar('cum_final_loss', self.cum_final_loss)
        tf.summary.scalar('cum_loss', self.cumulative_loss)

        self.train_step = tf.contrib.layers.optimize_loss(
            loss=self.cumulative_loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            optimizer='SGD',
            summaries=["loss", "gradient_norm"])

        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())
        self.trainable_variable_names = [v.name for v in tf.trainable_variables()]

    def double_predict_regression(self, response, confound_input, x_input):
        with tf.variable_scope('control_pred'):
            confound_preds, confound_loss = tf_utils.regressor(
                inputs=confound_input,
                labels=self.iter[response['name']],
                layers=self.params['regression_layers_1'],
                hidden=self.params['regression_hidden_1'],
                dropout=0.0)
            confound_preds = tf.expand_dims(confound_preds, 1)

        # force this into [batch size, attn width + 1]
        final_input = tf.concat([confound_preds, x_input], axis=1)
        final_input = tf.reshape(final_input, [-1, self.params['encoding_dim'] + 1])

        with tf.variable_scope('final_pred'):
            final_preds, final_loss = tf_utils.regressor(
                inputs=final_input,
                labels=self.iter[response['name']],
                layers=self.params['regression_layers_2'],
                hidden=self.params['regression_hidden_2'],
                dropout=0.0)

        return confound_preds, confound_loss, final_preds, final_loss


    def double_predict_classification(self, response, confound_input, x_input):
        with tf.variable_scope('control_pred'):
            confound_preds, confound_loss = tf_utils.classifier(
                inputs=confound_input,
                labels=self.iter[response['name']],
                layers=self.params['classification_layers_1'],
                num_classes=self.dataset.num_levels(response['name']),
                hidden=self.params['classification_hidden_1'],
                dropout=0.0,
                sparse_labels=True)


        final_input = tf.concat([confound_preds, x_input], axis=1)

        with tf.variable_scope('final_pred'):
            final_preds, final_loss = tf_utils.classifier(
                inputs=final_input,
                labels=self.iter[response['name']],
                layers=self.params['classification_layers_1'],
                num_classes=self.dataset.num_levels(response['name']),
                hidden=self.params['classification_hidden_2'],
                dropout=0.0,
                sparse_labels=True)

        return confound_preds, confound_loss, final_preds, final_loss


    def vectorize_confounds(self):
        # get all the controls into a vector:
        #  one-hot if categorical, carry through if scalar, 
        #  then put all those vecs tip to tip
        confounds = []
        for var in self.config.data_spec[1:]:
            if var['skip'] or not var['control']:
                continue
            if var['type'] == 'continuous':
                confounds.append(tf.expand_dims(self.iter[var['name']], 1))
            else:
                col_per_example = tf.expand_dims(self.iter[var['name']], 1)
                vecs = tf.map_fn(
                    lambda level: self._to_dense_vector(
                        level, self.dataset.num_levels(var['name'])),
                    col_per_example)
                confounds.append(tf.cast(vecs, tf.float32))
        confound_vecs = tf.concat(confounds, axis=1)
        return confound_vecs


    def _to_dense_vector(self, sparse_indices, total_features):
        descending_indices, _ = tf.nn.top_k(sparse_indices, k=tf.size(sparse_indices))
        ascending_indices = tf.reverse(descending_indices, axis=[0])
        unique_indices, _ = tf.unique(ascending_indices)
        vecs = tf.sparse_to_dense(
            sparse_indices=unique_indices,
            output_shape=[total_features],
            sparse_values=1)

        return vecs


    def train(self, sess):
        ops = [
            self.global_step,
            self.train_step,
            self.summaries
        ]
        return sess.run(ops)


    def test(self, sess):
        ops = [
            self.feature_weights,
            self.feature_intercept,
            self.final_output
        ]
        weights, intercept, output = sess.run(ops)

        output_scores = {varname: result['pred'] for varname, result in output.items()}

        feature_importance = {}
        for feature_name, weight in zip(open(self.dataset.vocab), weights):
            feature_importance[feature_name.strip()] = weight[0]
        feature_importance['intercept'] = intercept[0]

        return output_scores, feature_importance
