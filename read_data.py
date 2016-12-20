import numpy as np
import pickle
import os

def get_dialog_dict(train_file_loc,valid_file_loc,test_file_loc,vocab_size):
    '''Converts dialog file into list(.pkl) file where every entry in list represents one dialog and every dialog word is represented by its respective word id. 
    Args:
        train_file_loc:location of dailog file for training purpose.
        valid_file_loc:location of dailog file for validation purpose.
        test_file_loc:location of dailog file for testing purpose.
        vocab_size:number of vaocabulary words need to consider at time of training.'''

    train_script_run="python convert-text2dict.py "+train_file_loc+" --cutoff "+str(vocab_size)+" Training"
    valid_script_run="python convert-text2dict.py "+valid_file_loc+" --dict=Training.dict.pkl Validation"
    test_script_run="python convert-text2dict.py "+test_file_loc+" --dict=Training.dict.pkl Test"
    os.system(train_script_run)
    os.system(valid_script_run)
    os.system(test_script_run)

def get_utterances(ind1,utter_id):
    '''Retrieve three utterances from each dialog.
    Args:
        ind1: Index of entry 1 in dialog which specifies start and end of each utterance.
        utter_id: Array for each dialog having word id's for each word in dialog.
    Return: temp1: List of utterance1 word id's
            temp2: List of utterance2 word id's
            temp3: List of utterance3 word id's'''

    temp1=utter_id[ind1[0][0]:ind1[0][1]+1]
    temp1[len(temp1)-1]=2
    temp2=utter_id[ind1[0][1]:ind1[0][2]+1]
    temp2[len(temp2)-1]=2
    temp3=utter_id[ind1[0][2]:ind1[0][3]+1]
    temp3[len(temp3)-1]=2
    return temp1.tolist(),temp2.tolist(),temp3.tolist()

def get_utter_seq_len(dialogue_dict,max_len):
    '''Returns list of utterances and length of target utterance for batch of dialog data.
    Args:
        dialogue_dict: Dialog data where length of list is the number of dialogues and each entry has word id's for words of respective dialog. 
        max_len: max length of each utterance need to be considered for HRED model.
    Return:
        utter1_id: Word id's for words of utterance1 in a dialog. 
        utter2_id: Word id's for words of utterance2 in a dialog. 
        utter3_id: Word id's for words of utterance3 in a dialog. 
        target: Target word id's which will be 1 shifted than utter3_id for decoder purpose.
        decode_seq_len: Decoder input(utterance3) sequence length. '''

    utter1_id=[]
    utter2_id=[]
    utter3_id=[]
    target=[]
    decode_seq_len=[]
    for i in range(len(dialogue_dict)):
        utter_id=np.array(dialogue_dict[i])
        ind1=np.where(utter_id==1)
        #ind4=np.where(utter_id==4)
        temp1,temp2,temp3=get_utterances(ind1,utter_id)
        utter1_id.append(temp1)
        utter2_id.append(temp2)
        utter3_id.append(temp3)
        decode_seq_len.append(get_sequence_length(temp3,max_len))
        target.append(get_target(temp3))
    return utter1_id,utter2_id,utter3_id,target,decode_seq_len

def get_sequence_length(temp,max_len):
    '''Return length of target utterance for batch of dialog data.
    Args:
        temp:List of word id's for words of utterance3. 
        max_len: Max length of each utterance need to be considered for HRED model.
    Return:
         seq_len: Sequence length for each of decoder inputs(utterance3) '''

    seq_len=len(temp)
    if seq_len>max_len:
        seq_len=max_len
    return seq_len

def get_target(temp3):
    '''Returns target utterance which needs to be provided as decoder's output.
    Args:
        temp3: List of word id's for words of utterance3. 
    Returns: target which is a list of decoder output(true output) word id's.'''

    target=temp3[1:]
    return target

def pad_to_max(utterance_id,max_len):
    '''Performs padding upto maximum length for each utterance of dialog.
    Args:
        utterance_id: List of word id's for words of respective utterance.
        max_len: Max length of each utterance need to be considered for HRED model.
    Returns: utterance_id which is an array of size batch_size*max_len.'''

    for i in range(len(utterance_id)):
        if len(utterance_id[i])<max_len:
            to_append=max_len-len(utterance_id[i])
            for j in range(to_append):
                utterance_id[i].append(3)
        else:
            utterance_id[i]=utterance_id[i][0:max_len]
    return np.array(utterance_id)
             

def align_utterances(utter1,utter2,utter3,target,weights):
    '''Transpose utterance array and weight array into number of time steps*batch_size.'''    

    utter1=np.transpose(utter1)
    utter2=np.transpose(utter2)
    utter3=np.transpose(utter3)
    target=np.transpose(target)
    weights=np.transpose(weights)
    return utter1,utter2,utter3,target,weights

def get_weights(batch_target,seq_len):
    '''Creates a weight matrix for seq2seq.sequence_loss() function. 
    Args:
       batch_target: Target output of decoder of shape batch_size*max_len.
       seq_len: Sequence length for each of decoder inputs(utterance3)
    Return: weight matrix for batch size dialog data where each row of this matrix has 1's upto true length of target and remaining zeros upto maximum length.'''

    size1=batch_target.shape[0]
    size2=batch_target.shape[1]
    weights=np.zeros((size1,size2),dtype=np.float32)
    for i in range(len(seq_len)):
        weights[i][0:seq_len[i]]=1
    return weights

def get_batch(batch_dict,max_len):
    '''Retrieves utterances, performs padding upto maximum length and aligns each array as needed for HRED model for every batch sized dialog data.
    Args:
        batch_dict: Batch size dialog data at every iteration of training, validation and training. 
        max_len: Max length of each utterance need to be considered for HRED model.
    Return:
        batch_utter1: Batch size utterance1 data which is of shape batch_size*max_len for encoder of the model. 
        batch_utter2: Batch size utterance2 data which is of shape batch_size*max_len for encoder of the model. 
        batch_utter3: Batch size utterance3 data which is of shape batch_size*max_len for decoder(input) of the model. 
        batch_target: Batch size target data which is of shape batch_size*max_len for decoder(output) of the model. 
        batch_weights: Batch size weights need to be considerd for decoder(output) at the time of calculating seq2seq.sequence_loss() function. '''
    utter1_id,utter2_id,utter3_id,target,decode_seq_len=get_utter_seq_len(batch_dict,max_len)
    padded_utter1=pad_to_max(utter1_id,max_len)
    padded_utter2=pad_to_max(utter2_id,max_len)
    padded_utter3=pad_to_max(utter3_id,max_len)
    padded_target=pad_to_max(target,max_len)
    padded_weights=get_weights(padded_target,decode_seq_len)
    batch_utter1,batch_utter2,batch_utter3,batch_target,batch_weights=align_utterances(padded_utter1,padded_utter2,padded_utter3,padded_target,padded_weights)
    return batch_utter1,batch_utter2,batch_utter3,batch_target,batch_weights

def check_padding(dialog_data,param):
    '''Checks whether padding is required or not to make batch sized data at every time step.
    Args:
        dailog_data:List having Training or validation or test dialogue data.
        param:Parameter dictionary'''

    if len(dialog_data)%param['batch_size']!=0:
        pad_data(dialog_data,param)

def pad_data(dialog_data,param):
    ''' Call from check_padding function if padding is required to make batch sized data at every time step.
    Args:
        dailog_data:List having Training or validation or test dialogue data.
        param:Parameter dictionary'''

    batch_size=param['batch_size']
    rem_dialog=len(dialog_data)%batch_size
    append=batch_size-rem_dialog
    for i in range(append):
        append_data=[0 for k in range(param['n_speakers']*param['max_len'])] #Appending list of zeros(no of utterance*max_len times) so as to make batch sized data. 
        for j in range(0,len(append_data),param['max_len']/param['n_speakers']):
            append_data[j]=1
        dialog_data.append(append_data)
      
def load_valid_test_target(filename):
    '''Retrieves target utterance dialog from given filename which is used at the time of calculating bleu score for validation and test data.
    Args:
        filename: validation or test dialogue data file
    Return:sent_list which has target utterance(True output).'''

    sent_list=[]
    for line in open(filename,"r"):
        line=line.lower()
        ind=line.rindex('</s>')
        line=line[0:ind]
        line_words = line.strip().split('</s>')
        word_list=[]
        for words in line_words:
            if words!='':
                words=words.strip()
                word_list.append(words)
        sent_list.append(word_list[len(word_list)-1])    
    return sent_list
