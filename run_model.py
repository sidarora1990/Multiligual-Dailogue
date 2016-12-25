import math
import sys
import os
sys.path.append(os.getcwd())
import read_data
import math
import pickle
import argparse
import json
import random
import os.path
import shutil
import params
from params import *
import read_data
from read_data import *
from hierarchy_model import *
import nltk

def feeding_dict(model,inputs1,inputs2,decoder_data,true_op,weights,feed_prev):
    ''' Creates a mapping for placeholder inputs in form of feed_dict dictionary by providing each of them with respective list of input values.
    Args:
       model: Hierarchical Recurrent Encoder Decoder model object.
       inputs1: utterance1 input which is list of length param['max_len'] having tensor of size batch_size.
       inputs2: utterance2 input which is list of length param['max_len'] having tensor of size batch_size.
       decoder_data: decoder input(utterance3 without end symbol) which is list of length param['max_len'] having tensor of size batch_size.
       true_op: True output(utterance3 without start symbol) which is list of length param['max_len'] having tensor of size batch_size.
       weights: List of 1D batch-sized float-Tensors of the same length as above.
       feed_prev: Boolean; if True, only the first of decoder_inputs will be used (the "GO" symbol), and all other decoder inputs will be generated from previous decoder output.
    Return: feed_dict which is a dictionary which maps placeholders to their respective values.'''
       
    feed_dict={}
    for i,j in zip(model.encoder1_inputs,inputs1):
        feed_dict[i]=j
    for i,j in zip(model.encoder2_inputs,inputs2):
        feed_dict[i]=j
    for i,j in zip(model.decoder_inputs,decoder_data):
        feed_dict[i]=j
    for i,j in zip(model.targets,true_op):
        feed_dict[i]=j
    for i,j in zip(model.weights,weights):
        feed_dict[i]=j
    feed_dict[model.feed_previous]=feed_prev
    return feed_dict
    
def get_test_op(sess,model,batch_dict,param,logits,losses):
    '''Calculates loss and decoder output for test data.
    Args:
        sess: tf.Session() object created at the time of training.
        model: Hierarchical Recurrent Encoder Decoder model object.
        batch_dict: Batch sized dialogue data  for testing which is a list where every entry in list has word id's for words in a dialogue. 
        param: Parameter dictionary from params.py. Needed for fine tuning parameters of model.
        logits: Portion of the tf.Graph() where calculation of decoder's output is done.  
        losses: Portion of the tf.Graph() where loss calculation(sequence loss) is done.  
    Return:
        dec_op: List of length max_len having tensor of size batch_size*self.vocab_size.
        loss: A scalar float Tensor: average log-perplexity per symbol (weighted).'''
    
    cumulative_batch_loss=0
    batch_utter1,batch_utter2,batch_utter3,batch_target,batch_weights=read_data.get_batch(batch_dict,param['max_len'])
    feed_dict=feeding_dict(model,batch_utter1,batch_utter2,batch_utter3,batch_target,batch_weights,True)
    dec_op,loss=sess.run([logits,losses],feed_dict=feed_dict)
    for sentence_loss in loss:
        cumulative_batch_loss=cumulative_batch_loss+sentence_loss
    return dec_op,cumulative_batch_loss
    
def write_to_file(pred_op,true_op,current_dir):
    '''Writes predicted and true output(utterance3) for test data to file.
    Args:
        pred_op: decoder output for test data.
        true_op: true output(utterance3) for test data.
    '''
 
    pred_file=os.path.join(current_dir,'pred_op.txt') #Location of file where predicted output needs to be written.
    true_file=os.path.join(current_dir,'true_op.txt') #Location of file where true output needs to be written.
    f1=open(true_file,'w')
    for true_sentence in true_op:
        f1.write(str(true_sentence))
        f1.write('\n')
    f1.close()
    
    f2=open(pred_file,'w')
    for batch_sentence in pred_op:
        for pred_sentence in batch_sentence:
            f2.write(str(pred_sentence))
            f2.write('\n')
    f2.close()
    print "Test output written to file"

def perform_test(sess,model,saver,model_file,get_pred_sentence,param,logits,losses,current_dir):
    '''Perform decoding on test data by restoring already trained model.
    Args:
        sess: tf.Session() object created at the time of training.
        model: Hierarchical Recurrent Encoder Decoder model object.
        saver: Model saver object.
        model_file: Name of model file that needs to be restored using saver object.
        get_pred_sentence: function instance used at time of mapping word id's to respective words and then making it a sentence.
        param: Parameter dictionary from params.py. Needed for fine tuning parameters of model.
        logits: Portion of the tf.Graph() where calculation of decoder's output is done.  
        losses: Portion of the tf.Graph() where loss calculation(sequence loss) is done.  
        '''

    print "restoring the trained model"
    saver.restore(sess,model_file)
    test_dialogue_dict=pickle.load(open(param['test_dialogue_dict'],"rb"))
    read_data.check_padding(test_dialogue_dict,param)
    print "Test dialogues loaded"
    predicted_sentence=[]
    test_loss=0
    n_batches=len(test_dialogue_dict)/param['batch_size']
    true_sentence_list=read_data.load_valid_test_target(param['test_file_loc'])
    
    for i in range(n_batches):
        batch_dict=test_dialogue_dict[i*param['batch_size']:(i+1)*param['batch_size']]
        test_op,cumulative_batch_loss=get_test_op(sess,model,batch_dict,param,logits,losses)
        test_loss=cumulative_batch_loss+test_loss
        predicted_sentence.append(get_pred_sentence(test_op))

    pred_sent_wopad=predicted_sentence[0:len(true_sentence_list)]
    write_to_file(pred_sent_wopad,true_sentence_list,current_dir)
    print('average test loss is = %.6f:'%(float(test_loss)/n_batches))
    sys.stdout.flush()
    
def run_training(param,current_dir):
    ''' Performs training and validation on respective data. Contains different inner functions for performing them.
    Args:
        param: Parameter dictionary from params.py. Needed for fine tuning parameters of model.
    '''
        
    def get_train_loss(batch_dict):
        '''Calculates loss for training data. To avoid overfitting after 2 epochs feed_prev is set to true so that from now previous decoder time step output will be considered as input at current time step instead of considering gold input at every time step.
        Args:
            batch_dict: batch sized dialogue data for training.
        Return:
            loss: List of length max_len having tensor of size batch_size*self.vocab_size.
            dec_op: List of length max_len having tensor of size batch_size*self.vocab_size.
       '''
 
        batch_inputs1,batch_inputs2,batch_decoder_inputs,batch_targets,batch_weights=read_data.get_batch(batch_dict,param['max_len'])
        if epoch<=4:
            feed_dict=feeding_dict(model,batch_inputs1,batch_inputs2,batch_decoder_inputs,batch_targets,batch_weights,False)
        else:
            feed_dict=feeding_dict(model,batch_inputs1,batch_inputs2,batch_decoder_inputs,batch_targets,batch_weights,True)
        loss,dec_op,_=sess.run([losses,logits,train_op],feed_dict=feed_dict)
        return loss,dec_op
       
    def get_valid_loss(batch_dict):
        '''Calculates loss for validation data. Here feed_prev is always set to True so that previous decoder time step output will be considered as input to decoder at current time step for generating output.
        Args:
            batch_dict: batch sized dialogue data for training.
        Return:
            loss: List of length max_len having tensor of size batch_size*self.vocab_size.
            dec_op: List of length max_len having tensor of size batch_size*self.vocab_size.
        '''
        cumulative_batch_loss=0
        batch_utter1,batch_utter2,batch_utter3,batch_target,batch_weights=read_data.get_batch(batch_dict,param['max_len'])
        feed_dict=feeding_dict(model,batch_utter1,batch_utter2,batch_utter3,batch_target,batch_weights,True)
        loss,dec_op=sess.run([losses,logits],feed_dict=feed_dict)
        return loss,dec_op
    
    def get_cumulative_loss(batch_loss):   
        '''Return cumulative loss for batch of sentences
        Args:
            batch_loss: An array containing loss for every sentence from seq2seq.sequence_loss_by_example.
        Returns:
            cumulative_batch_loss: A scalar number giving cumulative loss for a particular batch.'''

        cumulative_batch_loss=0
        for sentence_loss in batch_loss:
            cumulative_batch_loss=cumulative_batch_loss+sentence_loss        
        return cumulative_batch_loss       
   
    def perform_training(batch_dict):
        '''Performs training over Train dataset
        batch_dict: Batch sized training data.
        Returns:
            cumulative_batch_loss: A scalar number giving cumulative loss for a particular batch of training data.
        '''
    
        batch_train_loss,dec_op=get_train_loss(batch_dict)
        cumulative_batch_loss=get_cumulative_loss(batch_train_loss)
        return cumulative_batch_loss 

    def perform_evaluation(batch_dict,batch_target_sentence,epoch,step):
        '''Performs evaluation on validation data.
        Args:
            batch_dict: Batch sized validation data.
            batch_target_sentence: True sentence(utterance3) for batch sized validation data.
            epoch: Training epoch 
            step: Training step
        Returns:
            cumulative_batch_loss: A scalar number giving cumulative loss for a particular batch of validation data.
        '''

        batch_valid_loss,valid_op=get_valid_loss(batch_dict) #Validation loss for a particular batch
        cumulative_batch_loss=get_cumulative_loss(batch_valid_loss)
        batch_predicted_sentence=get_pred_sentence(valid_op) 
        print_pred_true_op(batch_predicted_sentence,batch_target_sentence,step,epoch,batch_valid_loss) 
        return cumulative_batch_loss

    def evaluate(epoch,step):
        '''Performs evaluation on validation data.
        Args:
            epoch: Training epoch 
            step: Training step
        Return:
            valid_loss: Average validation loss.
        '''        

        print "Validation started"
        sys.stdout.flush()    
        valid_loss=0
        batch_predicted_sentence=[]
        n_batches=len(valid_dialogue_dict)/param['batch_size']
        for i in range(n_batches):
            start_index=i*param['batch_size']
            end_index=(i+1)*param['batch_size']
            batch_dict=valid_dialogue_dict[start_index:end_index] #Batch size validation data.
            if end_index>len(true_sentence_list):
                batch_target_sentence=true_sentence_list[start_index:len(true_sentence_list)]
            else:    
                batch_target_sentence=true_sentence_list[start_index:end_index] #Target output(dialogue) of decoder.
            cumulative_batch_loss=perform_evaluation(batch_dict,batch_target_sentence,epoch,step)
            valid_loss=valid_loss+cumulative_batch_loss
        return float(valid_loss)/len(valid_dialogue_dict) #Average validation loss across number of validation instances
       
    def print_pred_true_op(pred_op,true_op,step,epoch,batch_valid_loss):
        '''Prints predicted and true output for batch sized validation data.
        Args:
            pred_op: predicted output of decoder for batch sized validation data.
            true_op: true output for batch sized validation data.
            step: Training step
            epoch: Training epoch
        '''
        for i in range(len(true_op)):
            print "true sentence in step "+step+" of epoch "+epoch+" is:-"
            sys.stdout.flush()
            print true_op[i]
            sys.stdout.flush()
            print "\n"
            print "predicted sentence in step "+step+" of epoch "+epoch+" is:-"
            sys.stdout.flush()
            print pred_op[i]
            sys.stdout.flush()
            print "\n"
            print "loss for this pair of true and predicted sentence is "+str(batch_valid_loss[i])
            print "\n"
           

    def map_id_to_word(word_indices):
        ''' Maps predicted word id's to respective words and forms sentence for batch sized validation data.
        Args:
            word_indices: An array of shape batch_size*max_len having word id's for a particular batch.
        Return:
            sentence_list: List of predicted sentences for a batch of validation data.'''
        sentence_list=[]
        for sent in word_indices:
            word_list=[]
            for word_index in sent:
                word=train_dict[word_index] #Getting word from vocab dictionary for a particular index. 
                word_list.append(word) 
            sentence_list.append(" ".join(word_list)) #Forming sentence by joining words. 
        return sentence_list
    
    def get_pred_sentence(valid_op):
        '''Makes sentences from word id's for given batch size validation data.
        Args:
            valid_op: Decoder output for batch sized validation data.
        Return: 
            pred_sentence_list: List of batch sized predicted sentences.
        '''
 
        max_prob_index=[]
        for op in valid_op:
            max_index=np.argmax(op,axis=1)
            max_prob_index.append(max_index)
        max_prob_arr=np.transpose(max_prob_index)
        pred_sentence_list=map_id_to_word(max_prob_arr)
        return pred_sentence_list  

    train_dialogue_dict=pickle.load(open(param['train_dialogue_dict'],"rb"))
    print "Train dialogues loaded"
    read_data.check_padding(train_dialogue_dict,param)
    sys.stdout.flush()
    train_dict=pickle.load(open(param['train_dict'],"rb"))
    print "Vocab dictionary loaded"
    sys.stdout.flush()
    valid_dialogue_dict=pickle.load(open(param['valid_dialogue_dict'],"rb"))
    print "Validation dialogues loaded"
    sys.stdout.flush()
    read_data.check_padding(valid_dialogue_dict,param)
    true_sentence_list=read_data.load_valid_test_target(param['valid_file_loc'])
    print "target sentence list loaded"
    print "writing terminal output to file"
    f=open(param['valid_op'],'w')
    sys.stdout=f
    n_batches=len(train_dialogue_dict)/param['batch_size']
    model_file=os.path.join(current_dir,"best_model")
    with tf.Graph().as_default():
        #Creates an object for HRED model.
        model=Hierarchical_seq_model(param['embedding_size'],param['cell_size'],param['cell_type'],param['batch_size'],param['learning_rate'],param['epochs'],param['max_len'],param['patience'],param['decoder_words'],param['max_gradient_norm'],param['activation'])
        model.create_placeholder() 
        logits=model.inference() 
        losses=model.loss(logits)
        train_op=model.train(losses)
        print "model created"
        sys.stdout.flush()
        saver=tf.train.Saver()
        #tf.scalar_summary("loss",losses)
        #merged_summary=tf.merge_all_summaries()
        init=tf.initialize_all_variables()
        sess=tf.Session()
        if os.path.isfile(model_file):
            #Checking if model already exists.
            print "best model exists"
            saver.restore(sess,model_file) #Restoring existing model
        else:
        #summary_writer=tf.train.SummaryWriter(logs_path,graph=tf.get_default_graph())
            print "initializing fresh variables"
            sess.run(init)    
        best_valid_loss=float("inf")
        best_valid_epoch=0
        print "training started"
        sys.stdout.flush()
        for epoch in range(param['epochs']):
            
            train_loss=0
            for i in range(n_batches):
                batch_dict=train_dialogue_dict[i*param['batch_size']:(i+1)*param['batch_size']]
                cumulative_batch_loss=perform_training(batch_dict) 
                print('Epoch %d Step %d: train loss = %.6f' % (epoch,i, cumulative_batch_loss))
                sys.stdout.flush()
                train_loss=train_loss+cumulative_batch_loss
                #summary_writer.add_summary(summary,i)
            
                if i>0 and i%param['valid_freq']==0:
                    #Performing validation check.
                    valid_loss=evaluate(str(epoch),str(i))
                    print('Epoch %d Step %d: validation loss = %.6f' % (epoch,i,valid_loss))
                    sys.stdout.flush()
                    if best_valid_loss>valid_loss:
                        #Saving Model after comparing current validation loss with previous validation loss.
                        saver.save(sess,model_file)
                        best_valid_loss=valid_loss
                    #Beam Search
                else:
                    continue

            avg_train_loss=float(train_loss)/len(train_dialogue_dict) #Calculating average training loss
            print('Epoch %d: avg train loss mentioned epoch is= %.6f' % (epoch,avg_train_loss))
            random.shuffle(train_dialogue_dict) #Randomly shuffling training data after evry epoch. 
            valid_loss=evaluate(str(epoch),"last")
            print('Epoch %d: Validation loss for mentioned epoch is= %.6f' % (epoch,valid_loss))
            print "epoch "+str(epoch)+" of training is done"
            sys.stdout.flush()
            if best_valid_loss>valid_loss:
                saver.save(sess,model_file)
                best_valid_loss=valid_loss
                best_valid_epoch=epoch

            if epoch-best_valid_epoch>param['early_stop']:
                #Checking early stop condition. 
                print "results are not improving"
                break
                            
        print "Training over"
        print "Evaluation on Test data started"
        perform_test(sess,model,saver,model_file,get_pred_sentence,param,logits,losses,current_dir)   
    f.close()

def parse_args():
    '''Parsing arguments
    Parsing Arguments:
        prototype: Function in params.py, specifying which set of parameters need to use for this experiment. 
        output: Output directory
    '''
    parser=argparse.ArgumentParser()
    parser.add_argument("--prototype",type=str,help="Prototype to use must be specified")
    parser.add_argument("--output",type=str,help="Please specify directory for output files")
    args=parser.parse_args()
    return args

def set_path(current_param,current_dir):
    '''Set path of different vocab files required for training, validation and testing to respective output folder given parsing arguments.'''    
    current_param['train_dialogue_dict']=os.path.join(current_dir,current_param['train_dialogue_dict'])
    current_param['valid_dialogue_dict']=os.path.join(current_dir,current_param['valid_dialogue_dict'])
    current_param['test_dialogue_dict']=os.path.join(current_dir,current_param['test_dialogue_dict'])
    current_param['train_dict']=os.path.join(current_dir,current_param['train_dict'])
    current_param['valid_op']=os.path.join(current_dir,current_param['valid_op'])   
 
def get_file_list(file_list):
    '''Creates list of files(.pkl, model files, json files) present in a folder.
    Args:
        file_list: List of all files in a folder.
    Returns:
        pickle_list: List of pickle files present in a folder for a particular experiment.
        model_list: model file for a particular experiment.
        json_list: config json file for a particular experiment.
     '''
    pickle_list=[]
    model_list=[]
    json_list=[]
    for files in file_list:
        if files.endswith(".json"):
            json_list.append(files)
        elif files.startswith("best_model"):
            model_list.append(files)
        elif files.endswith(".pkl"):
            pickle_list.append(files)
    return json_list,pickle_list,model_list

def copy_files(source_dir,folder_files,current_dir):
    ''' Copy files from old folder to current(output) folder.
    Args:
        source_dir: Source directory.
        folder_files: Specific files of a folder.
        current_dir: Destination(output) directory.
    '''
    print "copying files"
    for files in folder_files:
        source_file=os.path.join(source_dir,files)
        if os.path.isfile(source_file):
            shutil.copy2(source_file,current_dir)

def check_prev_config(current_param,output_dir):
    ''''Checks whether new config file matches old config file. If it matches then copy different files from old folder to current(output) folder.
    current_param: current config file(json) for current set of parameters.
    output_dir: Output directory for dumping output files.
    '''

    old_file=0
    current_values=current_param.values()
    current_dir=output_dir
    for dirs in os.listdir(os.getcwd()):
        if dirs.startswith('output') and dirs!=current_dir:
            file_list=os.listdir(dirs)
            json_list,pickle_list,model_list=get_file_list(file_list)
            if len(json_list)!=0:
                old_json=os.path.join(dirs,json_list[0])
                old_param=json.load(open(old_json))
                old_values=old_param.values()
                # Checks whether vocab size, training, validation, testing files of old configuration are same as new configuration.
                if old_param["vocab_size"]==current_param["vocab_size"] and old_param['train_file_loc']==current_param['train_file_loc'] and old_param['valid_file_loc']==current_param['valid_file_loc'] and old_param['test_file_loc']==current_param['test_file_loc']:
                    print "config exists"
                    if len(pickle_list)!=0:
                        print "dictionary exists"
                        # If old config matches with new config file copy old vocab files to current(output) directory.
                        copy_files(dirs,pickle_list,current_dir)
                # Checks whether old configuration is completely same as new configuration file(json)
                if set(current_values)==set(old_values):
                    print "checking model"
                    # If condition holds load model files from old folder to current(output) folder.
                    if len(model_list)!=0:
                        print "model also exists"
                        old_file=1
                        copy_files(dirs,model_list,current_dir)
            if old_file==1:                
                break
            else:
                continue

def check_files_dirs(args):
    '''Checks for previous config files and set path accordingly for current set of parameters.
    Args:
        args: command line arguments.
    Returns:
        current_param: current set of parameters for a given experiment .
        output_dir: Output directory for dumping output files.
     '''

    output_dir=args.output
    current_param=eval(args.prototype)() 
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    param_copy=current_param.copy()
    param_copy['cell_type']=str(param_copy['cell_type'])
    param_file=os.path.join(output_dir,"config.json") 
    json.dump(param_copy,open(param_file,"wb"))
    check_prev_config(param_copy,output_dir)
    set_path(current_param,output_dir)
    return current_param,output_dir

def main():
    args = parse_args()
    current_param,current_dir=check_files_dirs(args)      
    if os.path.isfile(current_param['train_dialogue_dict']) and os.path.isfile(current_param['valid_dialogue_dict']) and os.path.isfile(current_param['train_dict']) and os.path.isfile(current_param['test_dialogue_dict']):
        print "all 4 dictionaries already exists"
        sys.stdout.flush()
    else:
        read_data.get_dialog_dict(current_param['train_file_loc'],current_param['valid_file_loc'],current_param['test_file_loc'],current_param['vocab_size'],current_dir)
        print "dictionary formed"
        sys.stdout.flush()
    run_training(current_param,current_dir)

if __name__=='__main__':
    main()	
