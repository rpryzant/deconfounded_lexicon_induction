
import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
from tensorflow.python.framework import function
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import numpy as np
from src.models.abstract_model import Model
import src.models.neural.tf_utils as tf_utils
import src.msc.utils as utils
import tf_utils

# # # # global gradient reversal functions  # # # #
def reverse_grad_grad(op, grad):
    return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
    return tf.identity(tensor)
# # # # # # # # # # # # # # # # # # # # # # # # # #





class A_BOW:
    """ A+BOW model """

    @staticmethod
    def build_model_graph(config, params, dataset):
        graph = tf.Graph()
        with graph.as_default():
            iterators = dataset.make_tf_iterators(params)
            model = A_BOW(config, params, dataset, iterators)

        return tf_utils.TFModel(graph=graph, model=model, iterator=iterators)


    def __init__(self, config, params, dataset, iterators):
        self.iter = iterators
        self.config = config
        self.params = params
        self.dataset = dataset

        self.learning_rate = tf.constant(params['learning_rate'])
        self.global_step = tf.Variable(0, trainable=False)

        source_name = dataset.input_varname()
        self.input_text, input_ids, input_lens = self.iter[source_name]

        # transform input text into big BOW vector
        with tf.variable_scope('input'):
            input_vector = tf.map_fn(
                lambda seq: self._to_dense_vector(seq, self.dataset.vocab_size),
                self.iter[dataset.input_varname()][1])
            input_encoded = tf_utils.fc_tube(
                inputs=tf.cast(input_vector, tf.float32),
                num_outputs=self.params['encoder_layers'],
                layers=self.params['encoder_layers'])
        # TODO this is PAINFULLY hacky!!!
        cur_graph = tf.get_default_graph()
        self.feature_weights = cur_graph.get_tensor_by_name(
            'input/layer_0/weights:0')
        self.feature_intercept = cur_graph.get_tensor_by_name(
            'input/layer_0/biases:0')


        # now build all the prediction heads
        self.step_output = defaultdict(dict)
        for variable in self.config.data_spec[1:]:
            if variable['skip']:
                continue

            with tf.variable_scope(variable['name'] + '_prediction_head'):
                if variable['control']:
                    prediction_input = self.reverse(input_encoded)
                else:
                    prediction_input = tf.identity(input_encoded)

                if variable['type'] == 'categorical':
                    preds, mean_loss = tf_utils.classifier(
                        inputs=prediction_input,
                        labels=self.iter[variable['name']],
                        layers=self.params['classifier_layers'],
                        num_classes=self.dataset.num_classes(variable['name']),
                        hidden=self.params['classifier_units'],
                        dropout=0.0,
                        sparse_labels=True)
                elif variable['type'] == 'continuous':
                    preds, mean_loss = tf_utils.regressor(
                        inputs=prediction_input,
                        labels=self.iter[variable['name']],
                        layers=self.params['regressor_layers'],
                        hidden=self.params['regressor_units'],
                        dropout=0.0)
                else:
                    raise Exception('ERROR: unknown type %s for variable %s' % (
                        variable['type'], variable['name']))

                mean_loss = tf.scalar_mul(variable['weight'], mean_loss)

            tf.summary.scalar('%s_loss' % variable['name'], mean_loss)
            self.step_output[variable['name']]['loss'] = mean_loss
            self.step_output[variable['name']]['pred'] = preds

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
        tf.summary.scalar('regularization_loss', reg_term)

        # now optimize
        self.loss = tf.reduce_sum([x['loss'] for x in self.step_output.values()])
        self.loss += reg_term
        tf.summary.scalar('global_loss', self.loss)

        self.train_step = tf.contrib.layers.optimize_loss(
            loss=self.loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            clip_gradients=self.params['gradient_clip'],
            optimizer='Adam',
            summaries=["gradient_norm"])

        # savers, summaries, etc
        self.trainable_variable_names = [v.name for v in tf.trainable_variables()]
        self.summaries = tf.summary.merge_all()
        self.saver = tf.train.Saver(tf.global_variables())


    def _to_dense_vector(self, sparse_indices, total_features):
        descending_indices, _ = tf.nn.top_k(sparse_indices, k=tf.size(sparse_indices))
        ascending_indices = tf.reverse(descending_indices, axis=[0])
        unique_indices, _ = tf.unique(ascending_indices)
        vecs = tf.sparse_to_dense(
            sparse_indices=unique_indices,
            output_shape=[total_features],
            sparse_values=1)

        return vecs


    def reverse(self, in_tensor):
        """ gradient reversal layer
        """
        input_shape = in_tensor.get_shape()
        out_tensor = reverse_grad(in_tensor)
        out_tensor.set_shape(input_shape)
        return out_tensor


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
            self.step_output
        ]
        weights, intercept, output = sess.run(ops)

        output_scores = {varname: result['pred'] for varname, result in output.items()}

        feature_importance = {}
        for feature_name, weight in zip(open(self.dataset.vocab), weights):
            feature_importance[feature_name.strip()] = weight[0]
        feature_importance['intercept'] = intercept[0]

        return output_scores, feature_importance
