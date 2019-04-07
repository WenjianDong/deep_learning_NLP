import keras.backend as K
from keras.layers import Layer
from keras import initializers, regularizers, constraints
import tensorflow as tf

def dot_product(x, kernel):
    """
    https://github.com/richliao/textClassifier/issues/13#issuecomment-377323318
    Wrapper for dot product operation, in order to be compatible with both
    Theano and Tensorflow
    Args:
        x (): input
        kernel (): weights
    Returns:
    """
    if K.backend() == 'tensorflow':
        return K.squeeze(K.dot(x, K.expand_dims(kernel)), axis=-1)
    else:
        return K.dot(x, kernel)

class AttentionWithMultipleContexts(Layer):
    """
    Attention operation, with a context/query matrix, for temporal data, as described in Lin et al. 2017 A structured self-attentive sentence embedding [https://arxiv.org/pdf/1703.03130.pdf]
    # Input shape
        3D tensor with shape: `(samples, steps, features)`.
    # Output shape
        3D tensor with shape: `(samples, number of context vectors, features)`.
    
    How to use:
    Just put it on top of an RNN Layer (GRU/LSTM/SimpleRNN) with return_sequences=True.
    The dimensions are inferred based on the output shape of the RNN.
    
    Note: The layer has been tested with Keras 2.2.0
    
    """
    
    def __init__(self,
                 W_regularizer=None, U_regularizer=None, b_regularizer=None,
                 W_constraint=None, U_constraint=None, b_constraint=None,
                 bias=True, nb_contexts=10, **kwargs):
        self.supports_masking = True
        self.nb_contexts = nb_contexts
        self.init = initializers.get('glorot_uniform')
        
        self.W_regularizer = regularizers.get(W_regularizer)
        self.U_regularizer = regularizers.get(U_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)
        
        self.W_constraint = constraints.get(W_constraint)
        self.U_constraint = constraints.get(U_constraint)
        self.b_constraint = constraints.get(b_constraint)
        
        self.bias = bias
        super(AttentionWithMultipleContexts, self).__init__(**kwargs)
    
    def build(self, input_shape):
        assert len(input_shape) == 3
        
        # for the shapes in comments: b denotes batch size (samples), i input size (steps), f the nb of features, and c the nb of context vectors
        self.W = self.add_weight((input_shape[-1], input_shape[-1]), # W shape: (f, f)
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        if self.bias:
            self.b = self.add_weight((input_shape[-1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        
        self.U = self.add_weight((input_shape[-1],self.nb_contexts), # U shape: (f,c)
                                 initializer=self.init,
                                 name='{}_U'.format(self.name),
                                 regularizer=self.U_regularizer,
                                 constraint=self.U_constraint)
        
        super(AttentionWithMultipleContexts, self).build(input_shape)
    
    def compute_mask(self, input, input_mask=None):
        # do not pass the mask to the next layers
        return None
    
    def call(self, x, mask=None):
        uit = dot_product(x, self.W) # (b,i,f) * (f,f) -> (b,i,f)
        
        if self.bias:
            uit += self.b
        
        uit = K.tanh(uit) # up to this point, not different from the single context self-attention
        
        A = K.dot(uit,self.U) # (b,i,f) * (f,c) -> (b,i,c)
        
        #A = K.exp(A) # element-wise application of the exponential function (doesn't change the shape)
        
        # apply mask after the exp. will be re-normalized next
        #if mask is not None:
        #    # Cast the mask to floatX to avoid float64 upcasting in theano
        #    A *= K.cast(mask, K.floatx())
        
        # in some cases especially in the early stages of training the sum may be almost zero
        # and this results in NaN's. A workaround is to add a very small positive number ε to the sum.
        # a /= K.cast(K.sum(a, axis=1, keepdims=True), K.floatx())
        #A /= K.cast(K.sum(A, axis=2, keepdims=True) + K.epsilon(), K.floatx()) # axis=1 as we are normalizing across the elements in the input (A is of shape (b,i,c)) # cast changes the type of the tensor to float
        
        #A = K.softmax(A,axis=1)
        A = tf.nn.softmax(A, dim=1)
        
        #A = K.expand_dims(A) # adds a batch dimension. remove?
        AT = K.permute_dimensions(A,(0,2,1)) # (b,c,i)
        weighted_input_mat = K.batch_dot(AT,x) # (b,c,i) * (b,i,f) -> (b,c,f) - final M matrix in the paper
        
        # for the custom loss
        #my_term = K.batch_dot(AT,K.permute_dimensions(AT,(0,2,1))) - K.eye(self.nb_contexts) # broadcasting is used for the difference
        
        return [weighted_input_mat, AT]
            
    def compute_output_shape(self, input_shape):
        return [(input_shape[0], self.nb_contexts, input_shape[-1]), (input_shape[0], self.nb_contexts, input_shape[1], )]

