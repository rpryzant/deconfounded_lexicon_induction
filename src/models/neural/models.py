"""
base model graph all others build off of
"""
import tensorflow as tf
from tensorflow.python.framework import function
from tensorflow.contrib.rnn.python.ops import rnn
from graph_module import GraphModule
import numpy as np
import encoders
import attention
import os



# # # # global gradient reversal functions  # # # #
def reverse_grad_grad(op, grad):
    return tf.constant(-1.) * grad


@function.Defun(tf.float32, python_grad_func=reverse_grad_grad)
def reverse_grad(tensor):
    return tf.identity(tensor)
# # # # # # # # # # # # # # # # # # # # # # # # # #







class Model:
    """
    multi-headed model for adversarial discriminative prediction
    """
    def __init__(self, config, sess, dataset):
        self.sess = sess

        # dataset config
        self.batch_size = config.batch_size
        self.max_len = config.max_len
        self.vocab_size = dataset.get_vocab_size()
        self.num_shops = dataset.get_num_shops()
        self.num_categories = dataset.get_num_categories()

        # model config
        self.embedding_size = config.embedding_size
        self.hidden_size = config.hidden_size
        self.num_layers = config.num_layers
        self.num_attention_units = config.num_attention_units
        self.prediction_hidden_size = config.prediction_hidden_size
        self.attention_type = config.attention_type
        self.attention_keys = config.attention_keys
        self.attention_order = config.attention_order

        # optimization config
        self.optimizer = config.optimizer
        self.gradient_clip = config.gradient_clip
        self.train_dropout = config.dropout_rate
        self.mixing_ratio = config.mixing_ratio
        self.reverse_gradients = config.reverse_gradients

        # training
        self.learning_rate = tf.placeholder(tf.float32, shape=(), name='lr')
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')

        # inputs
        self.source = tf.placeholder(tf.int32, [self.batch_size, self.max_len], name='source')
        self.source_len = tf.placeholder(tf.int32, [self.batch_size], name='source_len')

        # outputs
        self.log_sales = tf.placeholder(tf.float32, [self.batch_size], name='log_sales')
        self.shop = tf.placeholder(tf.int32, [self.batch_size], name='shop')
        self.price = tf.placeholder(tf.float32, [self.batch_size], name='price')
        self.category = tf.placeholder(tf.int32, [self.batch_size], name='category')

        # encode the source
        source_encoding = self.encode(self.source, self.source_len)

        # if config.attention_order == before_split, make prediction heads
        #   share the same attention graph
        # TODO -- this precludes tf from reading checkpoints for some 
        #          reason???
        attention_fn = None
        if self.attention_order == 'before_split':
            attention_fn = self.build_attention_fn()

        # run the encoding through each prediction head
        with tf.variable_scope('sales'):    # can also make a classifier
            self.sales_hat, self.sales_loss, self.sales_attn = self.regressor(
                self.log_sales,
                source_encoding, 
                reverse_grads=False, 
                attention_fn=attention_fn,
                name='sales')
        with tf.variable_scope('price'):
            self.price_hat, self.price_loss, self.price_attn = self.regressor(
                self.price, 
                source_encoding, 
                reverse_grads=self.reverse_gradients, 
                attention_fn=attention_fn,
                name='price')
        with tf.variable_scope('shop'):
            self.shop_hat, self.shop_loss, self.shop_attn = self.classifier(
                self.shop, 
                source_encoding, 
                num_classes=self.num_shops,
                reverse_grads=self.reverse_gradients, 
                attention_fn=attention_fn,
                name='shop')
        with tf.variable_scope('category'):
            self.category_hat, self.category_loss, self.category_attn = self.classifier(
                self.category, 
                source_encoding, 
                num_classes=self.num_categories,
                reverse_grads=self.reverse_gradients, 
                attention_fn=attention_fn,
                name='category')

        # get everything nice and tidy 
        self.loss = self.mixing_ratio * self.sales_loss + \
                ((1 - self.mixing_ratio) * self.price_loss + self.shop_loss + self.category_loss)
        self.train_step = self.optimize(self.loss)
        self.saver = tf.train.Saver()


    def train_on_batch(self, source, source_len, log_sales, price, shop, category, learning_rate=0.0003):
        """ train the model on a batch of data
        """
        _, sales_hat, price_hat, shop_hat, category_hat, loss = \
            self.sess.run([self.train_step, self.sales_hat, self.price_hat, self.shop_hat, self.category_hat, self.loss],
                                feed_dict={
                                    self.source: source,
                                    self.source_len: source_len,
                                    self.log_sales: log_sales,
                                    self.price: price,
                                    self.shop: shop,
                                    self.category: category,
                                    self.learning_rate: learning_rate,
                                    self.dropout: self.train_dropout
                                })
        return sales_hat, price_hat, shop_hat, category_hat, loss


    def test_on_batch(self, source, source_len, log_sales, price, shop, category):
        """ runs a forward pass on a batch's worth of data
        """
        sales_hat, price_hat, shop_hat, category_hat, loss, attn, a2, a3, a4 = \
            self.sess.run([self.sales_hat, self.price_hat, self.shop_hat,
                            self.category_hat, self.loss, self.sales_attn,
                            self.price_attn, self.shop_attn, self.category_attn],
                                feed_dict={
                                    self.source: source,
                                    self.source_len: source_len,
                                    self.log_sales: log_sales,
                                    self.price: price,
                                    self.shop: shop,
                                    self.category: category,
                                    self.dropout: 0.0
                                })        
        return sales_hat, price_hat, shop_hat, category_hat, loss, attn


    def save(self, path):
        """ saves model params at path specified by "path"
        """ 
        save_path = os.path.join(path, 'model')
        self.saver.save(self.sess, save_path, global_step=self.global_step)


    def load(self, filepath=None, dir=None):
        print('INFO: reading checkpoint...')
        if dir is not None:
            ckpt = tf.train.get_checkpoint_state(dir)
            if ckpt and ckpt.model_checkpoint_path:
                self.saver.restore(self.sess, ckpt.model_checkpoint_path)
                print 'INFO: success! model restored from %s' % ckpt.model_checkpoint_path
            else:
                raise Exception("ERROR: No checkpoint found at ", dir)
        elif filepath is not None:
            self.saver.restore(self.sess, filepath)
            print 'INFO: success! model restored from ', filepath
        else:
            raise Exception('ERROR: must provide a checkpoint filepath or directory')



    def optimize(self, loss):
        """ create a training op
        """
        train_op = tf.contrib.layers.optimize_loss(
            loss=loss,
            global_step=self.global_step,
            learning_rate=self.learning_rate,
            clip_gradients=self.gradient_clip,
            optimizer=self.optimizer,
            summaries=["learning_rate", "loss", "gradients", "gradient_norm"])
        return train_op


    def regressor(self, labels, encoder_output, reverse_grads=False, attention_fn=None, name='regressor'):
        """ attach a pair of fc layers to encoder_output and predict labels
            optionally reverse gradient flow into the encoder
        """
        # TODO - this is repeated wit hclassifier. find out if safe to abstract?
        encoder_output_output = encoder_output.outputs
        encoder_output_output_shape = encoder_output_output.get_shape()
        encoder_output_att_values = encoder_output.attention_values
        encoder_output_att_values_shape = encoder_output_att_values.get_shape()
        encoder_att_values_length = encoder_output.attention_values_length

        if reverse_grads:
            encoder_output_output = reverse_grad(encoder_output_output)
            encoder_output_output.set_shape(encoder_output_output_shape)
            encoder_output_att_values = reverse_grad(encoder_output_att_values)
            encoder_output_att_values.set_shape(encoder_output_att_values_shape)

        attention_fn = self.build_attention_fn() if attention_fn is None else attention_fn
        scores, attentional_context = attention_fn(
            query=tf.zeros_like(encoder_output_output[:, 0, :]),
            keys=encoder_output_output,
            values=encoder_output_att_values,
            values_length=encoder_att_values_length)

        # fc to hidden
        fc1 = tf.contrib.layers.fully_connected(
            inputs=attentional_context,
            num_outputs=self.prediction_hidden_size,
            activation_fn=tf.nn.relu,
            scope='%s_fc' % name)  
        # fc to preds
        preds = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=1,
            activation_fn=None,
            scope='%s_pred' % name)
        preds = tf.squeeze(preds)

        # mean per-batch l2 loss
        loss = tf.nn.l2_loss(preds - labels)
        loss = loss / self.batch_size  # mean per-example loss

        return preds, loss, scores


    def classifier(self, labels, encoder_output, num_classes, reverse_grads=False, attention_fn=None, name='classifier'):
        """ attach a pair of fc layers to encoder_output and predict labels
            optionally reverse gradient flow into the encoder
        """
        # TODO - this is repeated with regressor. find out if safe to abstract?
        encoder_output_output = encoder_output.outputs
        encoder_output_output_shape = encoder_output_output.get_shape()
        encoder_output_att_values = encoder_output.attention_values
        encoder_output_att_values_shape = encoder_output_att_values.get_shape()
        encoder_att_values_length = encoder_output.attention_values_length

        if reverse_grads:
            encoder_output_output = reverse_grad(encoder_output_output)
            encoder_output_output.set_shape(encoder_output_output_shape)
            encoder_output_att_values = reverse_grad(encoder_output_att_values)
            encoder_output_att_values.set_shape(encoder_output_att_values_shape)

        attention_fn = self.build_attention_fn() if attention_fn is None else attention_fn
        scores, attentional_context = attention_fn(
            query=tf.zeros_like(encoder_output_output[:, 0, :]),
            keys=encoder_output_output,
            values=encoder_output_att_values,
            values_length=encoder_att_values_length)

        # fc to hidden
        fc1 = tf.contrib.layers.fully_connected(
            inputs=attentional_context,
            num_outputs=self.prediction_hidden_size,
            activation_fn=tf.nn.relu,
            scope='%s_fc' % name)  
        # fc to logits
        logits = tf.contrib.layers.fully_connected(
            inputs=fc1,
            num_outputs=num_classes,
            activation_fn=None,
            scope='%s_pred' % name)

        # mean log perplexity per batch
        losses = tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels)
        mean_loss = tf.reduce_mean(losses)

        return logits, mean_loss, scores


    def encode(self, source, source_len):
        """ run the source through an encoder
        """
        with tf.variable_scope('embedding'):
            source_embedded = self.get_embeddings(self.source)
        with tf.variable_scope('encoder'):
            encoder_output = self.run_encoder(source_embedded, self.source_len)

        return encoder_output


    def get_embeddings(self, source):
        """ looks up word embeddings for a source sequence
        """
        E = tf.get_variable('E',
                            shape=[self.vocab_size, self.embedding_size])
        embedding = tf.nn.embedding_lookup(E, source)
        return embedding


    def run_encoder(self, source, source_len):
        """ runs the source embeddings through an encoder
        """
        cell1, cell2 = self.build_rnn_cells()

        if self.attention_keys == 'word_vectors':
            encoder = encoders.IdentityEncoder()
        elif self.num_layers == 1:
            encoder = encoders.BidirectionalEncoder(cell1, cell2)
        else:
            encoder = encoders.StackedBidirectionalEncoder(cell1, cell2)

        encoder_output = encoder(source, source_len)
        return encoder_output


    def build_attention_fn(self):
        if self.attention_type == 'bahdanau':
            attention_fn = attention.AttentionLayerBahdanau(num_units=self.num_attention_units)
        elif self.attention_type == 'dot':
            attention_fn = attention.AttentionLayerDot()
        elif self.attention_type == 'fc':
            attention_fn = attention.AttentionLayerFc()
        else:
            raise Exception("unrecognized attention type provided! ", self.attention_type)
        return attention_fn


    def build_rnn_cells(self):
        """ builds an rnn cell
        """
        cell1 = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell1 = tf.contrib.rnn.DropoutWrapper(cell1,
                                             input_keep_prob=(1 - self.dropout))
        cell2 = tf.contrib.rnn.BasicLSTMCell(self.hidden_size, state_is_tuple=True)
        cell2 = tf.contrib.rnn.DropoutWrapper(cell2,
                                             input_keep_prob=(1 - self.dropout))

        if self.num_layers == 1:
            return cell1, cell2

        stacked_cell = tf.contrib.rnn.MultiRNNCell([cell] * self.num_layers,
                                                   state_is_tuple=True)
        return stacked_cell









