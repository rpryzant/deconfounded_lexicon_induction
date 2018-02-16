import tensorflow as tf
from collections import namedtuple
from graph_module import GraphModule
from tensorflow.contrib.rnn.python.ops import rnn



EncoderOutput = namedtuple(
    "EncoderOutput",
    "outputs final_state attention_values attention_values_length")


class StackedBidirectionalEncoder(GraphModule):
    """ multi-layer bidirectional encoders
    """
    def __init__(self, cell, name='stacked_bidirectional'):
        super(StackedBidirectionalEncoder, self).__init__(name)
        self.cell = cell

    def _build(self, inputs, lengths):
        outputs, final_fw_state, final_bw_state = rnn.stack_bidirectional_dynamic_rnn(
            cells_fw=self.cell._cells,
            cells_bw=self.cell._cells,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)

        # Concatenate states of the forward and backward RNNs
        final_state = final_fw_state, final_bw_state

        return EncoderOutput(
            outputs=outputs,
            final_state=final_state,
            attention_values=outputs,
            attention_values_length=lengths)


class BidirectionalEncoder(GraphModule):
    """ single-layer bidirectional encoder
    """
    def __init__(self, cell1, cell2, name='bidirectional'):
        super(BidirectionalEncoder, self).__init__(name)
        self.cell1 = cell1
        self.cell2 = cell2

    def _build(self, inputs, lengths):
        outputs_pre, final_state = tf.nn.bidirectional_dynamic_rnn(
            cell_fw=self.cell1,
            cell_bw=self.cell2,
            inputs=inputs,
            sequence_length=lengths,
            dtype=tf.float32)

        # Concatenate outputs of the forward and backward RNNs
        outputs = tf.concat(outputs_pre, 2)

        return EncoderOutput(
            outputs=outputs,
            final_state=final_state,
            attention_values=outputs,
            attention_values_length=lengths)


class IdentityEncoder(GraphModule):
    """ do-nothing encoder
    """
    def __init__(self, name='identity'):
        super(IdentityEncoder, self).__init__(name)


    def _build(self, inputs, lengths):
        return EncoderOutput(
            outputs=inputs,
            final_state=tf.zeros_like(inputs[:,0,:]),
            attention_values=inputs,
            attention_values_length=lengths)








