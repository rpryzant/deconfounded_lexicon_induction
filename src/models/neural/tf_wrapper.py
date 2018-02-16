import sys
sys.path.append('../..')
import os
import time
from collections import namedtuple, defaultdict
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

from src.models.abstract_model import Model, Prediction
import src.models.neural.A_BOW as A_BOW
import src.models.neural.A_ATTN as A_ATTN
import src.models.neural.DR_BOW as DR_BOW
import src.models.neural.DR_ATTN as DR_ATTN
import src.msc.utils as utils

import numpy as np


class TFModelWrapper(Model):
    """ base wrapper that manipulates each tensorflow graph
    """
    def __init__(self, config, params):
        self.config = config
        self.params = params


    def save(self, model_dir):
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        self.loaded_model.model.saver.save(
            self.sess,
            os.path.join(model_dir, 'model.ckpt'),
            global_step=self.global_step)
        print 'TF WRAPPER: saved into ', model_dir


    def load(self, dataset, model_dir):
        self.loaded_model, self.global_step, self.sess = \
            self.create_or_load_model(model_dir, dataset)
        return self.loaded_model


    def create_or_load_model(self, model_dir, dataset):
        """ not refactored into self.load() because of shared code paths
        """
        latest_ckpt = tf.train.latest_checkpoint(model_dir)

        model = self.model_builder_class.build_model_graph(
            self.config, self.params, dataset)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth=True
        sess = tf.Session(graph=model.graph, config=config)

        with model.graph.as_default():
            tf.set_random_seed(self.config.seed)

            if latest_ckpt:
                start_time = time.time()
                model.model.saver.restore(sess, latest_ckpt)
                sess.run(tf.tables_initializer())
                print "INFO: loaded model parameters from %s, time %.2fs" % \
                    (latest_ckpt, time.time() - start_time)
            else:
                start_time = time.time()
                sess.run(tf.global_variables_initializer())
                sess.run(tf.tables_initializer())
                print "INFO: created model with fresh parameters, time %.2fs" % \
                                (time.time() - start_time)

        print "INFO: trainable variables:"
        values = sess.run(model.model.trainable_variable_names)
        for name, value in zip(model.model.trainable_variable_names, values):
            print '\t%s   ----    Shape: %s' %  (name, value.shape)

        global_step = model.model.global_step.eval(session=sess)
        return model, global_step, sess      


    def train(self, dataset, model_dir):
        self.loaded_model, self.global_step, self.sess = \
            self.create_or_load_model(model_dir, dataset)
        summary_writer = tf.summary.FileWriter(
            os.path.join(model_dir, "train_log"), self.loaded_model.graph)

        self.sess.run(self.loaded_model.model.iter['initializer'])

        epochs = 0
        start_time = time.time()
        prog = utils.Progbar(target=self.params['num_train_steps'])
        while self.global_step < self.params['num_train_steps']:
            try:
                self.global_step, loss, summary = self.loaded_model.model.train(self.sess)
                prog.update(self.global_step, [('train loss', loss)])
                summary_writer.add_summary(summary, self.global_step)
            except tf.errors.OutOfRangeError:
                epochs += 1
#                print 'epoch ', epochs, ' took %.2fs' % (time.time() - start_time)
                start_time = time.time()
                self.sess.run(self.loaded_model.model.iter['initializer'])


    def inference(self, dataset, model_dir, dev=True):
        self.sess.run(self.loaded_model.model.iter['initializer'])

        all_feature_importance = defaultdict(list)
        predictions = {}
        try:
            while True:
                scores, feature_importance = self.loaded_model.model.test(self.sess)

                for response, scores in scores.items():
                    if response not in predictions:
                        predictions[response] = scores
                    else:
                        predictions[response] = np.concatenate(
                            (predictions[response], scores), axis=0)

                for feature, value in feature_importance.items():
                    all_feature_importance[feature].append(value)

        except tf.errors.OutOfRangeError:
            pass

        if 'mean' in self.params.get('attn_importance_strategy', 'mean'):
            feature_importance = {k: np.mean(v) for k, v in all_feature_importance.items()}
        elif 'max' in self.params['attn_importance_strategy']:
            feature_importance = {k: np.max(v) for k, v in all_feature_importance.items()}
        else:
            raise Exception("attention importance strategy unknown: %s" % self.params['attn_importance_strategy'])
        return Prediction(
            scores=predictions,
            feature_importance=feature_importance)

    def report(self):
        """ releases self.report, a summary of the last job this model
                executed whether that be training, testing, etc
        """
        # TODO
        raise NotImplementedError


class ABOWWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = A_BOW.A_BOW

class AATTNWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = A_ATTN.A_ATTN

class DRBOWWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = DR_BOW.DR_BOW

class DRATTNWrapper(TFModelWrapper):
    def __init__(self, config, params):
        TFModelWrapper.__init__(self, config, params)

        self.model_builder_class = DR_ATTN.DR_ATTN





