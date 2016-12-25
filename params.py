import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
import os
def prototype_1():
    param={}
    param['train_file_loc']="data/enfr_train_50_40.txt"
    param['valid_file_loc']="data/enfr_valid_50_50.txt"
    param['test_file_loc']="data/enfr_test_50_50.txt"
    param['train_dialogue_dict']='Training.dialogues.pkl'
    param['valid_dialogue_dict']='Validation.dialogues.pkl'
    param['test_dialogue_dict']='Test.dialogues.pkl'
    param['train_dict']='Train.dict.pkl'
    param['valid_op']="val_op.txt"
    param['cell_size']=512
    param['cell_type']=rnn_cell.GRUCell
    param['batch_size']=64
    param['vocab_size']=100000
    param['embedding_size']=256
    param['learning_rate']=0.0004
    param['patience']=200
    param['early_stop']=10
    param['epochs']=10
    param['max_len']=20
    param['valid_freq']=5000
    param['max_gradient_norm']=0.1
    param['activation']=None
    param['n_speakers']=3
    param['decoder_words']=param['vocab_size']+3#unk and pad symbol
    return param

def prototype_2():
    param={}
    param['train_file_loc']="data/valid_5.txt"
    param['valid_file_loc']="data/valid_5.txt"
    param['test_file_loc']="data/valid_5.txt"
    param['train_dialogue_dict']='Training.dialogues.pkl'
    param['valid_dialogue_dict']='Validation.dialogues.pkl'
    param['test_dialogue_dict']='Test.dialogues.pkl'
    param['train_dict']='Train.dict.pkl'
    param['valid_op']="val_op.txt"
    param['cell_size']=20
    param['cell_type']=rnn_cell.GRUCell
    param['batch_size']=5
    param['vocab_size']=80
    param['embedding_size']=20
    param['learning_rate']=0.004
    param['patience']=200
    param['early_stop']=10
    param['epochs']=3
    param['max_len']=20
    param['valid_freq']=1
    param['max_gradient_norm']=0.1
    param['activation']=None
    param['n_speakers']=3
    param['decoder_words']=param['vocab_size']+3#unk and pad symbol
    return param
