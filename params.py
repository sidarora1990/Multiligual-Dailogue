import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
def get_params():
    param={}
    param['train_file_loc']="./valid_5.txt"
    param['valid_file_loc']="./valid_5.txt"
    param['test_file_loc']="./valid_5.txt"
    param['train_dialogue_dict']='./Training.dialogues.pkl'
    param['valid_dialogue_dict']='./Validation.dialogues.pkl'
    param['test_dialogue_dict']='./Test.dialogues.pkl'
    param['train_dict']='./Train.dict.pkl'
    param['valid_op']="./val_op2.txt"
    param['cell_size']=20
    param['cell_type']=rnn_cell.BasicRNNCell
    param['batch_size']=2
    param['vocab_size']=80
    param['embedding_size']=20
    param['learning_rate']=0.002
    param['patience']=200
    param['early_stop']=10
    param['epochs']=2
    param['max_len']=20
    param['valid_freq']=1
    param['max_gradient_norm']=0.1
    param['activation']=None
    param['n_speakers']=3
    param['decoder_words']=param['vocab_size']+3#unk and pad symbol
    param['model_path']="./model"
    param['logs_path']="./logs"
    return param
