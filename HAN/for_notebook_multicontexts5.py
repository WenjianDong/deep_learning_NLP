# This file is composed mainly by Antoine Tixier. I made changes on several details. 

import sys
import os
import json
import numpy as np

from gensim.models import KeyedVectors

from keras.models import Model
from keras import optimizers, backend as K
from keras.backend.tensorflow_backend import _to_tensor
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Lambda, Input, Embedding, Dropout, Dense, TimeDistributed, Reshape

from keras.losses import categorical_crossentropy

import time

# = = = = = high-level parameters
is_GPU = True
dataset_name = 'amazon_review_full_csv'

multicontext = True
is_lr_range_test = True
is_grid_search = True

#The following information is based on experiments:
#max_lr = 0.34826174
#base_lr = max_lr/6

print('dataset:',dataset_name)
print('multicontext?',multicontext)
print('LR range test?',is_lr_range_test)
print('grid search?',is_grid_search)

# = = = = = model parameters

if dataset_name == 'imdb-review-dataset':
    n_cats = 2
    n_units = 30 # IMDB is much smaller than Amazon, so we need a less complex model and more regularization
    drop_rate = 0.55
elif dataset_name == 'amazon_review_full_csv':
    n_cats = 5
    n_units = 50
    drop_rate = 0.45

max_doc_size_overall = 20 # max nb of sentences allowed per document
max_sent_size_overall = 50 # max nb of words allowed per sentence    

my_nb_contexts_sents = 3

# replace with your own!
if is_GPU:
    path_root = '/home/wenjian/data/' + dataset_name + '/'

    path_to_batches = path_root + 'batches/'

    path_to_functions = '/home/wenjian/codes/deep_learning_NLP/HAN/'

    path_to_save = '/home/wenjian/results/' + dataset_name + '/'

else:
    path_root = 'H:/' + dataset_name + '/'
    path_to_batches = path_root + 'batches/'
    path_to_functions = 'C:/Users/mvazirg/Dropbox/deep_learning_NLP/HAN/'

# custom classes and functions
sys.path.insert(0, path_to_functions)
from AttentionWithMultipleContexts import AttentionWithMultipleContexts
from AttentionWithContext import AttentionWithContext
from CyclicalLearningRate import CyclicLR
from CyclicalMomentum import CyclicMT
from han_my_functions import read_batches, bidir_gru, PerClassAccHistory, LossHistory, AccHistory, LRHistory, MTHistory

# = = = = = = = = = = = 

#for my_lambda in [round(elt,2) for elt in np.linspace(0,1,10)]: 

my_lambda = 0.2
print('my_lambda: ', my_lambda)
def ortho_check(my_tensor,my_nb_contexts_sents=my_nb_contexts_sents,my_lambda=my_lambda):
    '''
    my_tensor's shape: (batch size, my_nb_contexts_sents, nb of words in sent)
    returns (batch size, 1)
    '''
    my_diff = K.batch_dot(my_tensor,K.permute_dimensions(my_tensor,(0,2,1))) - K.eye(my_nb_contexts_sents) # (b,c,s)*(b,s,c) -> (b,c,c) # the difference is broadcasted to all elts in batch
    return my_lambda * K.sqrt(K.mean(K.square(my_diff),axis=(1,2)) + K.epsilon()) # we sum each flattened matrix and return batch-size floats (we add the K.epsilon() to avoid getting inf values)

ortho_check_layer = Lambda(ortho_check)

def custom_loss_wrapper(my_tensors):
    '''
    based on the accepted answer here: https://stackoverflow.com/questions/46464549/keras-custom-loss-function-accessing-current-input-pattern
    my_tensors' shape: (doc size, batch size, nb of contexts, sent size) - in practice my_tensors will be the matrices of attentional coefficients, each one of shape (batch size, nb of contexts, sent size)
    TimeDistributed iterates by default over index 1 - but here we want to iterate over the sents 
    hence the permute_dimensions so that 'my_tensors' becomes (batch size, nb of sents, nb of contexts, nb of words in each sent)
    TimeDistributed(ortho_check_layer) returns my_nb_contexts_sents float values for each sent. We average them and end up with one value for each elt in the batch
    '''
    def custom_loss(y_true, y_pred):
        # for debugging:  K.eval(K.categorical_crossentropy(K.variable([[1,0,2],[1,0,2]]),K.variable([[2,0,2],[1,1,2]])))
        return K.categorical_crossentropy(y_true, y_pred) + K.sum(TimeDistributed(ortho_check_layer)(my_tensors),axis=1)
        #return K.categorical_crossentropy(y_true, y_pred) + K.sum(TimeDistributed(ortho_check_layer)(K.permute_dimensions(my_tensors,(1,0,2,3))),axis=0) #  permuted my_tensors is of shape (batch size, doc size, nb of contexts, sent size) - TimeDistributed(ortho_check_layer) returns (batch size, doc_size, 1) and we compute the mean of the doc_size floats -> (batch size, 1)
    return custom_loss

gensim_obj = KeyedVectors.load(path_root + 'word_vectors.kv', mmap='r') # needs an absolute path!
word_vecs = gensim_obj.wv.syn0
# add Gaussian initialized vector on top of embedding matrix (for padding)
pad_vec = np.random.normal(size=word_vecs.shape[1]) 
word_vecs = np.insert(word_vecs,0,pad_vec,0)


# The metrics to record the original loss and the regularization loss separately. 
def custom_metrics_wrapper_pure_loss(my_tensors):
    def custom_metrics_pure_loss(y_true, y_pred):
        return K.categorical_crossentropy(y_true, y_pred) # We don't need to sum or average across different samples my ourselves. Keras does that for us. 
    return custom_metrics_pure_loss

def custom_metrics_wrapper_regularization_loss(my_tensors):
    def custom_metrics_regularization_loss(y_true, y_pred):
        return K.sum(TimeDistributed(ortho_check_layer)(my_tensors),axis=1)
    return custom_metrics_regularization_loss



# for debugging
#word_vecs = np.zeros((1000,100))

if multicontext:

    # = = = = = sent encoder = = = = = = = =
    sent_ints = Input(shape=(None,))
    sent_wv = Embedding(input_dim=word_vecs.shape[0],
                        output_dim=word_vecs.shape[1],
                        weights=[word_vecs],
                        input_length=None, # sentence size varies from batch to batch
                        trainable=True
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU) # annotations for each word in the sentence

    sent_att_mat, AT = AttentionWithMultipleContexts(nb_contexts=my_nb_contexts_sents)(sent_wa)

    sent_att_vec = Reshape((n_units*2*my_nb_contexts_sents,))(sent_att_mat)

    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)

    sent_encoder = Model(inputs=sent_ints,outputs=[sent_att_vec_dr, AT])

    # = = = = = doc encoder = = = = = = = =

    doc_ints = Input(shape=(None,None,))        
    #sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)

    # the below is based on: https://github.com/keras-team/keras/issues/6449#issuecomment-298255231
    outputs = []
    for out in sent_encoder.output:
        outputs.append(TimeDistributed(Model(sent_encoder.input,out))(doc_ints))

    sent_att_vecs_dr, ATs = outputs # there are as many ATs as sentences

    doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU) # annotations for each sentence in the document

    doc_att_vec,_ = AttentionWithContext(return_coefficients=True)(doc_sa) # attentional vector for the document
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

    #my_nb_contexts_docs = 5
    #doc_att_mat,doc_coeffs_mat = AttentionWithMultipleContexts(nb_contexts=my_nb_contexts_docs)(doc_sa)
    #doc_att_vec = Reshape((n_units*2*my_nb_contexts_docs,))(doc_att_mat)
    #doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

    preds = Dense(units=n_cats,
                  activation='softmax')(doc_att_vec_dr)

    han = Model(doc_ints,preds)

    my_loss = custom_loss_wrapper(ATs)

    met_regu = custom_metrics_wrapper_regularization_loss(ATs)
    met_pure = custom_metrics_wrapper_pure_loss(ATs)

    print('multicontext model defined')

    # for debugging
    '''
    han.compile(loss=my_loss,
            optimizer='adam',
            metrics=['accuracy']) 
    X = np.array([[[0]*10]]) #np.random.randint(2,10,(1,1,10))
    y = np.random.randint(1,5,(1,5))
    han.train_on_batch(X,y)

    get_ats = Model(doc_ints,ATs)
    ATs = get_ats.predict(X)

    K.eval(ortho_check(_to_tensor(ATs[0,:,:,:],dtype='float32')))

    custom_loss_wrapper(_to_tensor(ATs[0,:,:,:],dtype='float32'))

    K.eval(K.sqrt(K.sum(K.square(_to_tensor(np.zeros((1,10,10)),dtype='float32')),axis=(1,2))))

    A = _to_tensor(np.array([np.zeros((10,5)),np.ones((10,5))]),dtype='float32')

    K.eval(K.softmax(A,axis=2))

    my_denominator = K.cast(K.sum(A, axis=2, keepdims=True) + K.epsilon(), K.floatx())

    A = A /my_denominator

    AT = K.permute_dimensions(A,(0,2,1))

    K.mean(_to_tensor(np.array([[0],[0],[0]]),dtype='float32'))

    K.eval(K.softmax(_to_tensor(np.array([[0,1],[0,0],[0,0]]),dtype='float32')))

    K.mean(TimeDistributed(ortho_check_layer)(K.permute_dimensions(my_tensors,(1,0,2,3))

    '''

else:

    sent_ints = Input(shape=(None,))
    sent_wv = Embedding(input_dim=word_vecs.shape[0],
                        output_dim=word_vecs.shape[1],
                        weights=[word_vecs],
                        input_length=None, # sentence size vary from batch to batch
                        trainable=True
                        )(sent_ints)

    sent_wv_dr = Dropout(drop_rate)(sent_wv)
    sent_wa = bidir_gru(sent_wv_dr,n_units,is_GPU) # annotations for each word in the sentence
    sent_att_vec, sent_att_coeffs = AttentionWithContext(return_coefficients=True)(sent_wa) # attentional vector for the sentence
    sent_att_vec_dr = Dropout(drop_rate)(sent_att_vec)                      
    sent_encoder = Model(sent_ints,sent_att_vec_dr)

    doc_ints = Input(shape=(None,None,))        
    sent_att_vecs_dr = TimeDistributed(sent_encoder)(doc_ints)
    doc_sa = bidir_gru(sent_att_vecs_dr,n_units,is_GPU) # annotations for each sentence in the document
    doc_att_vec, doc_att_coeffs = AttentionWithContext(return_coefficients=True)(doc_sa) # attentional vector for the document
    doc_att_vec_dr = Dropout(drop_rate)(doc_att_vec)

    preds = Dense(units=n_cats,
                  activation='softmax')(doc_att_vec_dr)

    han = Model(doc_ints,preds)

    my_loss = 'categorical_crossentropy'

    print('single context model defined')


# ============================ training

if is_lr_range_test:
    is_grid_search = False

batch_names = os.listdir(path_to_batches)

if is_grid_search or is_lr_range_test:
    batch_names_train = [elt for elt in batch_names if 'train_' in elt]
    batch_names_val = [elt for elt in batch_names if 'val_' in elt] 
else:
    batch_names_train = [elt for elt in batch_names if 'train_' in elt or 'val_' in elt]
    batch_names_val = [elt for elt in batch_names if 'test_' in elt]

its_per_epoch_train = int(len(batch_names_train)/2) # /2 because there are batch files for documents and labels
its_per_epoch_val = int(len(batch_names_val)/2)

rd_train = read_batches(batch_names_train,
                        path_to_batches,
                        do_shuffle=True,
                        do_train=True,
                        my_max_doc_size_overall=max_doc_size_overall,
                        my_max_sent_size_overall=max_sent_size_overall,
                        my_n_cats=n_cats)

rd_val = read_batches(batch_names_val,
                      path_to_batches,
                      do_shuffle=False,
                      do_train=True,
                      my_max_doc_size_overall=max_doc_size_overall,
                      my_max_sent_size_overall=max_sent_size_overall,
                      my_n_cats=n_cats)

'''
# for debugging
# = = = checking that the model works in non-batch mode: = = =
my_batch = rd_train.__next__()
my_docs = my_batch[0]
docs_ints = my_docs[:1,:,:]
K.eval(han(K.variable(docs_ints)))
'''

if is_lr_range_test:
    nb_epochs = 6
    my_patience = nb_epochs
    step_size = its_per_epoch_train*nb_epochs
    base_lr, max_lr = 0.001, 1

else:
    max_lr =  0.34826174 # add here the values returned by the LR range test or good priors
    base_lr = max_lr/6
    nb_epochs = 30
    half_cycle = 6 
    my_patience = half_cycle*2
    step_size = its_per_epoch_train*half_cycle


#The following information is based on experiments:
#max_lr = 0.34826174
#base_lr = max_lr/6


print(its_per_epoch_train,its_per_epoch_val,step_size)

base_mt, max_mt = 0.85, 0.95

my_optimizer = optimizers.SGD(lr=base_lr,
                              momentum=max_mt, # we decrease momentum when lr increases
                              decay=1e-5,
                              nesterov=True)

han.compile(loss=my_loss,
            optimizer=my_optimizer,
            metrics=['accuracy', met_regu, met_pure])    

# quick sanity check
#han.fit(x=my_batch[0],y=my_batch[1])

lr_sch = CyclicLR(base_lr=base_lr, max_lr=max_lr, step_size=step_size, mode='triangular')
mt_sch = CyclicMT(base_mt=base_mt, max_mt=max_mt, step_size=step_size, mode='triangular')

early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as accuracy on validation set increases
                               patience=my_patience,
                               mode='max')

# make sure that the model corresponding to the best epoch is saved
checkpointer = ModelCheckpoint(filepath=path_to_save + 'han_trained_weights',
                               monitor='val_acc',
                               save_best_only=True,
                               mode='max',
                               verbose=0,
                               save_weights_only=True) # so that we can train on GPU and load on CPU (for CUDNN GRU)

# batch-based callbacks
loss_hist = LossHistory()
acc_hist = AccHistory()
lr_hist = LRHistory()
mt_hist = MTHistory() 

if not (is_grid_search or is_lr_range_test):
    callback_list = [loss_hist,acc_hist,lr_hist,mt_hist,lr_sch,mt_sch,early_stopping,checkpointer]
else:
    callback_list = [loss_hist,acc_hist,lr_hist,mt_hist,lr_sch,mt_sch,early_stopping] # don't save weights during the grid search (would use too much space)

start_time = time.time()

han.fit_generator(rd_train, 
                  steps_per_epoch=its_per_epoch_train, 
                  epochs=nb_epochs,
                  callbacks=callback_list,
                  validation_data=rd_val, 
                  validation_steps=its_per_epoch_val,
                  use_multiprocessing=False, 
                  workers=1)

end_time = time.time()


hist = han.history.history

if is_lr_range_test: # we need to record batch-level metrics only for the LR range test
    hist['batch_loss'] = loss_hist.loss_avg
    hist['batch_acc'] = acc_hist.acc_avg
    hist['batch_lr'] = lr_hist.lrs
    hist['batch_mt'] = mt_hist.mts

hist = {k:[str(elt) for elt in v] for k, v in hist.items()}

#with open(path_to_save + 'han_hist_singlectx.json', 'w') as file:
    #json.dump(hist, file, sort_keys=False, indent=4)

#with open(path_to_save + 'han_hist_gridsearch10_lambda=' + str(my_lambda) + '.json', 'w') as file:
    #json.dump(hist, file, sort_keys=False, indent=4)

with open(path_to_save + 'han_hist_normal.json', 'w') as file:
    json.dump(hist, file, sort_keys=False, indent=4)


#print('= = = = =',my_lambda,'done = = = = =')

#K.clear_session()
#print('keras session cleared')

# To record the key parameters of the experiment
with open(path_to_save + 'time_log.txt', 'w') as f:
    f.write('my_lambda: ' + str(my_lambda) + '\n')
    f.write('total training time: ' + str(end_time-start_time) + '\n')
    f.write('multicontexts: ' + str(multicontext) + '\n')
    f.write('context number: ' +str(my_nb_contexts_sents) + '\n')
    f.write('dataset name: ' + str(dataset_name) + '\n')
    f.write('is_lr_range_test: ' + str(is_lr_range_test) + '\n')
    f.write('is_grid_search: ' + str( is_grid_search) + '\n')
    f.write('epoch number: '+ str(nb_epochs) + '\n')
    f.write('is_lr_range_test: ' + str(is_lr_range_test) + '\n') 
    f.write('is_grid_search: ' + str(is_grid_search) + '\n')


