# encoding: utf-8


import sys
sys.path.append('..') 
import time

import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn as bi_rnn
from sklearn.linear_model import LogisticRegression

from src.classify import Classifier, read_node_label
from modules import *
from hyperparams import Hyperparams as hp
import pickle



import evaluate_graph_reconstruction as gr
import evaluate_link_prediction as lp
import evaluation_util
import evaluate_node_classification as ncf
import networkx as nx
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.manifold import TSNE


from numpy import linalg as la

#import matplotlib.pyplot as pltl




class STNE(object):
    def __init__(self, hidden_dim, node_num, fea_dim, seq_len, depth=1, node_fea=None, node_fea_trainable=False, is_training=True):
        self.node_num, self.fea_dim, self.seq_len = node_num, fea_dim, seq_len

        self.input_seqs = tf.placeholder(tf.int32, shape=(None, self.seq_len), name='input_seq')
        
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        if node_fea is not None:
            assert self.node_num == node_fea.shape[0] and self.fea_dim == node_fea.shape[1]
            self.embedding_W = tf.Variable(initial_value=node_fea, name='encoder_embed', trainable=node_fea_trainable)
        else:
            self.embedding_W = tf.Variable(initial_value=tf.random_uniform(shape=(self.node_num, self.fea_dim)),
                                           name='encoder_embed', trainable=node_fea_trainable)
        self.input_seq_embed = tf.nn.embedding_lookup(self.embedding_W, self.input_seqs, name='input_embed_lookup')   ##shape(?,10,1433)
        
        
        
        
        
        
        
        
        
        
        ####zuo yi ge adj_fea
        
        
        ######encode
#        self.input_seq_embed_lear = tf.layers.dense(self.input_seq_embed, units=hp.hidden_units, activation=tf.nn.relu,name='dense')
        self.enc = tf.layers.dropout(self.input_seq_embed, 
                                    rate=hp.dropout_rate, 
                                    training=tf.convert_to_tensor(is_training))
        
        
        
        
        
        
        
        
        
        
        
        with tf.variable_scope("encoder"):


            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ### Multihead Attention
                    self.enc = multihead_attention(queries=self.enc, 
                                                    keys=self.enc, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=False)
                    
                    ### Feed Forward
                    self.enc = feedforward(self.enc, num_units=[4*hp.hidden_units, hp.hidden_units])        
        
        
        
        
        
        
        #####decode
        with tf.variable_scope("decoder"):
            ## Embedding
            self.dec = embedding(self.input_seqs, 
                                  vocab_size=self.node_num, 
                                  num_units=hp.hidden_units,
                                  scale=True, 
                                  scope="dec_embed")
            
            
            self.dec = tf.layers.dropout(self.dec, 
                                        rate=hp.dropout_rate, 
                                        training=tf.convert_to_tensor(is_training))
        
        
        

            for i in range(hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    ## Multihead Attention ( self-attention)
                    self.dec = multihead_attention(queries=self.dec, 
                                                    keys=self.dec, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads, 
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training,
                                                    causality=True, 
                                                    scope="self_attention")
                    
                    ## Multihead Attention ( vanilla attention)
                    self.dec = multihead_attention(queries=self.dec, 
                                                    keys=self.enc, 
                                                    num_units=hp.hidden_units, 
                                                    num_heads=hp.num_heads,
                                                    dropout_rate=hp.dropout_rate,
                                                    is_training=is_training, 
                                                    causality=False,
                                                    scope="vanilla_attention")
                    
                    ## Feed Forward
                    self.dec = feedforward(self.dec, num_units=[4*hp.hidden_units, hp.hidden_units])
        
        
        
        
        
        
        
        
        
        
        
        
        


                    
                    
                    
                    
                    
                    
                    
                    
        self.output_preds = tf.layers.dense(self.dec, units=self.node_num, activation=None)


#        self.pp=tf.nn.softmax(self.output_preds,axis=-1) ############




        self.preds = tf.to_int32(tf.arg_max(self.output_preds, dimension=-1))
        self.istarget = tf.to_float(tf.not_equal(self.input_seqs, 0))
        self.acc = tf.reduce_sum(tf.to_float(tf.equal(self.preds, self.input_seqs))*self.istarget)/ (tf.reduce_sum(self.istarget))
        tf.summary.scalar('acc', self.acc)        
        self.y_smoothed = label_smoothing(tf.one_hot(self.input_seqs, depth=node_num))
#        self.y_smoothed = tf.one_hot(self.input_seqs, depth=node_num)
        
        
        
#        self.loss_ce=tf.reduce_sum(tf.square(tf.subtract(self.y_smoothed,self.pp)))###################
        
        self.loss = tf.nn.softmax_cross_entropy_with_logits(logits=self.output_preds, labels=self.y_smoothed)
        self.loss_ce = tf.reduce_sum(self.loss*self.istarget) / (tf.reduce_sum(self.istarget))        
        
        
        
        
        
        
#        loss_ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.input_seqs, logits=self.output_preds)######3#####
#        self.loss_ce = tf.reduce_mean(loss_ce, name='loss_ce')
        
        
        
        self.global_step = tf.Variable(1, name="global_step", trainable=False)


def read_node_features(filename):
    fea = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        fea.append(np.array([float(x) for x in vec[1:]]))
    fin.close()
    return np.array(fea, dtype='float32')


def read_node_sequences(filename):
    seq = []
    fin = open(filename, 'r')
    for l in fin.readlines():
        vec = l.split()
        seq.append(np.array([int(x) for x in vec]))
    fin.close()
    return np.array(seq)


def reduce_seq2seq_hidden_mean(seq, seq_h, node_num, seq_num, seq_len):
    node_dict = {}
    for i_seq in range(seq_num):
        for j_node in range(seq_len):
            nid = seq[i_seq, j_node]
            if nid in node_dict:
                node_dict[nid].append(seq_h[i_seq, j_node, :])
            else:
                node_dict[nid] = [seq_h[i_seq, j_node, :]]
    vectors = []
    for nid in range(node_num):
        vectors.append(np.average(np.array(node_dict[nid]), 0))
    return np.array(vectors)


def reduce_seq2seq_hidden_add(sum_dict, count_dict, seq, seq_h_batch, seq_len, batch_start):
    for i_seq in range(seq_h_batch.shape[0]):
        for j_node in range(seq_len):
            nid = seq[i_seq + batch_start, j_node]
            if nid in sum_dict:
                sum_dict[nid] = sum_dict[nid] + seq_h_batch[i_seq, j_node, :]
            else:
                sum_dict[nid] = seq_h_batch[i_seq, j_node, :]
            if nid in count_dict:
                count_dict[nid] = count_dict[nid] + 1
            else:
                count_dict[nid] = 1
    return sum_dict, count_dict


def reduce_seq2seq_hidden_avg(sum_dict, count_dict, node_num):
    vectors = []
    for nid in range(node_num):
        vectors.append(sum_dict[nid] / count_dict[nid])
    return np.array(vectors)


def node_classification(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.enc,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.enc,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_enc.astype('float32'), seq_len, s_idx)
    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)
    
    
    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return (f1_micro,f1_macro),node_enc_mean


def node_classification_d(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    node_cnt = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_dec = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: len(sequences)], seqne.dropout: 0})
        enc_sum_dict, node_cnt = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt, sequences,
                                                           batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt, node_num=node_n)

    lr = Classifier(vectors=node_enc_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def node_classification_hd(session, bs, seqne, sequences, seq_len, node_n, samp_idx, label, ratio):
    enc_sum_dict = {}
    dec_sum_dict = {}
    node_cnt_enc = {}
    node_cnt_dec = {}
    s_idx, e_idx = 0, bs
    while e_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)
        s_idx, e_idx = e_idx, e_idx + bs

    if s_idx < len(sequences):
        batch_enc = session.run(seqne.encoder_output,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        enc_sum_dict, node_cnt_enc = reduce_seq2seq_hidden_add(enc_sum_dict, node_cnt_enc, sequences,
                                                               batch_enc.astype('float32'), seq_len, s_idx)

        batch_dec = session.run(seqne.decoder_outputs,
                                feed_dict={seqne.input_seqs: sequences[s_idx: e_idx], seqne.dropout: 0})
        dec_sum_dict, node_cnt_dec = reduce_seq2seq_hidden_add(dec_sum_dict, node_cnt_dec, sequences,
                                                               batch_dec.astype('float32'), seq_len, s_idx)

    node_enc_mean = reduce_seq2seq_hidden_avg(sum_dict=enc_sum_dict, count_dict=node_cnt_enc, node_num=node_n)
    node_dec_mean = reduce_seq2seq_hidden_avg(sum_dict=dec_sum_dict, count_dict=node_cnt_dec, node_num=node_n)

    node_mean = np.concatenate((node_enc_mean, node_dec_mean), axis=1)
    lr = Classifier(vectors=node_mean, clf=LogisticRegression())
    f1_micro, f1_macro = lr.split_train_evaluate(samp_idx, label, ratio)
    return f1_micro


def check_all_node_trained(trained_set, seq_list, total_node_num):
    for seq in seq_list:
        trained_set.update(seq)
    if len(trained_set) == total_node_num:
        return True
    else:
        return False
    
    
    
def get_batch_data(X,Y):
    # Load data
    #X, Y = load_train_data()
    
    # calc total batch count
    num_batch = len(X) // hp.batch_size
    
    # Convert to tensor
    X = tf.convert_to_tensor(X, tf.int32)
    Y = tf.convert_to_tensor(Y, tf.int32)
    
    # Create Queues
    input_queues = tf.train.slice_input_producer([X, Y])
            
    # create batch queues
    x, y = tf.train.shuffle_batch(input_queues,
                                num_threads=8,
                                batch_size=hp.batch_size, 
                                capacity=hp.batch_size*64,   
                                min_after_dequeue=hp.batch_size*32, 
                                allow_smaller_final_batch=False)
    
    return x, y, num_batch # (N, T), (N, T), ()
    
    
    
    
    
    


if __name__ == '__main__':
    
    name='cora'
    
    folder = '../data2/'
    fn = '../result/'+name+'_result.txt'
    
    
#    folder = '../data/cora/'
#    fn = '../data/cora/result.txt'

    dpt = 1            # Depth of both the encoder and the decoder layers (MultiCell RNN)
    h_dim = 256        # Hidden dimension of encoder LSTMs
    s_len = 10         # Length of input node sequence
    epc = 2            # Number of training epochs
    trainable = False  # Node features trainable or not
    dropout = 0.2      # Dropout ration
#    clf_ratio = [0.1, 0.2, 0.3, 0.4, 0.5]  # Ration of training samples in subsequent classification
    clf_ratio = [0.5]
    b_s = 256          # Size of batches
    lr = 0.1         # Learning rate of RMSProp

    start = time.time()
    fobj = open(fn, 'w')
    
    
    
    
#    X, Y = read_node_label(folder + 'labels.txt')
#    node_fea = read_node_features(folder + 'cora.features')
#    node_seq = read_node_sequences(folder + 'node_sequences_10_10.txt')
    
    
    all_data=pickle.load(open(folder+name+'.txt', 'rb'))
    X=all_data['X']
    Y=all_data['Y']
    node_fea1=all_data['node_fea']
    node_seq=all_data['node_seq']
    
    
    node_fea1=np.array(node_fea1)
    U, S, VT = la.svd(node_fea1)
    Ud = U[:, 0:h_dim]
    Sd = S[0:h_dim]
    node_fea = np.array(Ud)*Sd.reshape(h_dim)
    
    
    
    
    
    
    
    
    
    

    with tf.Session() as sess:
        model = STNE(hidden_dim=h_dim, node_fea_trainable=trainable, seq_len=s_len, depth=dpt, node_fea=node_fea,
                     node_num=node_fea.shape[0], fea_dim=node_fea.shape[1])
        
#        train_op = tf.train.AdagradOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)
        
        
        
#        train_op = tf.train.GradientDescentOptimizer(lr).minimize(model.loss_ce, global_step=model.global_step)
        train_op = tf.train.MomentumOptimizer(lr,momentum=0.9).minimize(model.loss_ce, global_step=model.global_step)



  
        sess.run(tf.global_variables_initializer())

        trained_node_set = set()
        all_trained = False
        all_f1=[]
        all_MAP_list=[]
        for epoch in range(epc):
            start_idx, end_idx = 0, b_s
            print('Epoch,\tStep,\tLoss,\tacc, #Trained Nodes')
            while end_idx < len(node_seq):
                _, loss, step,acc = sess.run([train_op, model.loss_ce, model.global_step,model.acc], feed_dict={
                    model.input_seqs: node_seq[start_idx:end_idx], model.dropout: dropout})
#            sess.run(tf.get_default_graph().get_tensor_by_name('dense/kernel:0'))
                if not all_trained:
                    all_trained = check_all_node_trained(trained_node_set, node_seq[start_idx:end_idx],
                                                         node_fea.shape[0])

                if step % 1 == 0:
                    print(epoch, '\t', step, '\t', loss,'\t', acc, '\t', len(trained_node_set))
#                    if all_trained:
#                        f1_mi = []
#                        for ratio in clf_ratio:
#                            f1_mi.append(node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq,
#                                                             seq_len=s_len, node_n=node_fea.shape[0], samp_idx=X,
#                                                             label=Y, ratio=ratio))
#
#                        print('step ', step)
#                        fobj.write('step ' + str(step) + ' ')
#                        for f1 in f1_mi:
#                            print(f1)
#                            fobj.write(str(f1) + ' ')
#                        fobj.write('\n')
                start_idx, end_idx = end_idx, end_idx + b_s

            if start_idx < len(node_seq):
                sess.run([train_op, model.loss_ce, model.global_step, model.acc],
                         feed_dict={model.input_seqs: node_seq[start_idx:len(node_seq)], model.dropout: dropout})

            minute = np.around((time.time() - start) / 60)
            print('\nepoch ', epoch, ' finished in ', str(minute), ' minutes\n')
            
            
            f1_mi = []
            for ratio in clf_ratio:
                f1,Vecs=node_classification(session=sess, bs=b_s, seqne=model, sequences=node_seq, seq_len=s_len,
                                    node_n=node_fea.shape[0], samp_idx=X, label=Y, ratio=ratio)                
                f1_mi.append(f1)
            all_f1.append(f1_mi)

####################################
            
            
            
            
            

#            lb=np.transpose(np.array(lb))
        lb=Y
        mi=0
        ma=0
        for k in range(10):
            micro, macro = ncf.evaluateNodeClassification(Vecs, lb, 0.5)     ##test_ratio
            mi=mi+micro
            ma=ma+macro
        all_MAP_list.append([mi/10, ma/10])            
        
        
        
        
        
        
        
        
        
        fobj.write(str(epoch) + ' ')
        print('Classification results on current ')
        for f1 in f1_mi:
            print(f1)
            fobj.write(str(f1) + ' ')
        fobj.write('\n')
        minute = np.around((time.time() - start) / 60)

        fobj.write(str(minute) + ' minutes' + '\n')
        print('\nClassification finished in ', str(minute), ' minutes\n')
        
            
            
            
            
            
            

        fobj.close()
        minute = np.around((time.time() - start) / 60)
        print('Total time: ' + str(minute) + ' minutes')

