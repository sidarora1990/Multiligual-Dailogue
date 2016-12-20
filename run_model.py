import math
import sys
import os
sys.path.append(os.getcwd())
import read_data
import math
import pickle
import random
import os.path
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
    
def check_dir(param):
    '''Checks whether model and logs directtory exists, if not then creates both directories for saving best model and logs.
    Args:
        param:parameter dictionary.'''

    if not os.path.exists(param['logs_path']):
        os.makedirs(param['logs_path'])    
    if not os.path.exists(param['model_path']):
        os.makedirs(param['model_path'])

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
    
def write_to_file(pred_op,true_op):
    '''Writes predicted and true output(utterance3) for test data to file.
    Args:
        pred_op: decoder output for test data.
        true_op: true output(utterance3) for test data.
    '''
 
    pred_file='./pred_op.txt' #Location of file where predicted output needs to be written.
    true_file='./true_op.txt' #Location of file where true output needs to be written.
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

def perform_test(sess,model,saver,model_file,get_pred_sentence,param,logits,losses):
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
    write_to_file(pred_sent_wopad,true_sentence_list)
    print('average test loss is = %.6f:'%(float(test_loss)/n_batches))
    sys.stdout.flush()
    
def run_training(param):
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
        #if epoch<=3:
        feed_dict=feeding_dict(model,batch_inputs1,batch_inputs2,batch_decoder_inputs,batch_targets,batch_weights,False)
        #else:
        #    feed_dict=feeding_dict(model,batch_inputs1,batch_inputs2,batch_decoder_inputs,batch_targets,batch_weights,True)
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
    check_dir(param)      
    n_batches=len(train_dialogue_dict)/param['batch_size']
    model_file=os.path.join(param['model_path'],"best_model")
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
        print "all tensorflow variables are"
        all_var=tf.all_variables()
        print len(all_var)
        for var in all_var:
            print var.name
            print(var.get_shape())
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
                    #print('Epoch %d Step %d:Average Bleu score at validation time is = %6f:'%(epoch,i,avg_bleu_score))
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
        perform_test(sess,model,saver,model_file,get_pred_sentence,param,logits,losses)   
    f.close()

def main():
    param=params.get_params() #Retrieving parameter dictionary from params.py. Needed for fine tuning parameters of model.
    if os.path.isfile(param['train_dialogue_dict']) and os.path.isfile(param['valid_dialogue_dict']) and os.path.isfile(param['train_dict']):
        print "dictionary already exists"
        sys.stdout.flush()
    else:
        read_data.get_dialog_dict(param['train_file_loc'],param['valid_file_loc'],param['test_file_loc'],param['vocab_size'])
        print "dictionary formed"
        sys.stdout.flush()
    run_training(param)

if __name__=='__main__':
    main()
