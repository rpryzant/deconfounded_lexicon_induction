import tensorflow as tf
from graph_module import GraphModule
from tensorflow.python.framework import function 

class AttentionLayer(GraphModule):
    """ base layer for attention functionality
    """
    def __init__(self, num_units=128, name='attention'):
        GraphModule.__init__(self, name)
        self.num_units = num_units

    def score_fn(self, keys, query):
        """Computes the attention score"""
        raise NotImplementedError

    def _build(self, query, keys, values, values_length):
        """ computes attentional scores and weighted average of 
                encoder hidden states 
        """
        values_depth = values.get_shape().as_list()[-1]

        att_keys = tf.contrib.layers.fully_connected(
            inputs=keys,
            num_outputs=self.num_units,
            activation_fn=None,
            scope="att_keys")

        att_query = tf.contrib.layers.fully_connected(
            inputs=query,
            num_outputs=self.num_units,
            activation_fn=None,
            scope="att_query")

        scores = self.score_fn(att_keys, att_query)

        # Replace all scores for padded inputs with tf.float32.min
        num_scores = tf.shape(scores)[1]
        scores_mask = tf.sequence_mask(
            lengths=tf.to_int32(values_length),
            maxlen=tf.to_int32(num_scores),
            dtype=tf.float32)
        scores = scores * scores_mask + ((1.0 - scores_mask) * tf.float32.min)

        # Normalize the scores
        scores_normalized = tf.nn.softmax(scores, name="scores_normalized")

        # Calculate the weighted average of the attention inputs
        # according to the scores
        context = tf.expand_dims(scores_normalized, 2) * values
        context = tf.reduce_sum(context, 1, name="context")
        context.set_shape([None, values_depth])


        return (scores_normalized, context)




class AttentionLayerFc(AttentionLayer):
    """ runs the keys through an a pair of fc layers
    """
    def score_fn(self, keys, query):
        fc = tf.contrib.layers.fully_connected(
            inputs=keys,
            num_outputs=self.num_units,
            activation_fn=tf.nn.relu,
            scope="att_hidden")

        scores = tf.contrib.layers.fully_connected(
            inputs=fc,
            num_outputs=1,
            activation_fn=None,
            scope='att_score')

        return tf.squeeze(scores)


class AttentionLayerDot(AttentionLayer):
    """ attention according to https://arxiv.org/abs/1508.04025
    """
    def score_fn(self, keys, query):
        """Calculates a batch- and timweise dot product"""
        return tf.reduce_sum(keys * tf.expand_dims(query, 1), [2])

@function.Defun(
    tf.float32,
    tf.float32,
    tf.float32,
    func_name="att_sum_bahdanau",
    noinline=True)
def att_sum_bahdanau(v_att, keys, query):
    """Calculates a batch- and timweise dot product with a variable"""
    return tf.reduce_sum(v_att * tf.tanh(keys + tf.expand_dims(query, 1)), [2])


class AttentionLayerBahdanau(AttentionLayer):
    """An attention layer that calculates attention scores using
        a parameterized multiplication."""
    def score_fn(self, keys, query):
        v_att = tf.get_variable(
            "v_att", shape=[self.num_units], dtype=tf.float32)

        return att_sum_bahdanau(v_att, keys, query)




