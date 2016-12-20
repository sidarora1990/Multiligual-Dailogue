import tensorflow as tf
from tensorflow.python.ops import rnn,rnn_cell
from tensorflow.python.ops import control_flow_ops
#from tensorflow.python.ops import seq2seq
import seq2seq
from seq2seq import *
import numpy as np
import math
import sys
import os
sys.path.append(os.getcwd())
from seq2seq import *
class Hierarchical_seq_model():
    def __init__(self,embedding_size,cell_size,cell_type,batch_size,learning_rate,epochs,max_len,patience,decoder_words,max_gradient_norm,activation):
        '''Parameter initialization '''
        self.embedding_size=embedding_size
        #self.dec_input_size=dec_input_size
        self.cell_size=cell_size
        self.cell_type=cell_type
        self.batch_size=batch_size
        self.learning_rate=tf.Variable(float(learning_rate),trainable=False)
        self.epochs=epochs
        self.max_len=max_len#maximum length considered for encoder and decoder input
        self.patience=patience#early stopping criteria
        self.decoder_words=decoder_words#No of decoder words considered
        self.max_gradient_norm=max_gradient_norm#Gradient clipping criteria
        self.enc_layers=2#2 layers(word level and utterance level encoder).
        self.activation=activation
        self.encoder_inputs=[]#List for encoder inputs where length of list is self.time_steps.
        self.decoder_inputs=[]#List of decoder inputs where length of list is self.time_steps/2.
        self.targets=[]#List of true output values where length of list is self.time_steps/2. 
        self.weights=[]#List of weight values for calculating weighted sequence loss.
        self.enc_cells=[]#List for encoder cells(word and utterance level).
        self.dec_cells=[]#List for different decoder cells(different languages) for now only one.
        self.enc_scopes=[]#List for different scopes for word and utterance encoder.
        self.dec_scopes=[]#List for different scopes at decoder level if needed(future).

        def create_cell_scopes():
            '''Creating different cells and their scopes for encoder(word level encoder) and decoder. Scopes are different at word and utterance level encoder(No parameter sharing).'''
            for i in range(self.enc_layers):
                if i==self.enc_layers-1:
                    #Bidirectional RNN at utterance level, forward and backward cell.
                    self.enc_cells.append([self.cell_type(self.cell_size),self.cell_type(self.cell_size)])
                else:
                    self.enc_cells.append(self.cell_type(self.cell_size))
                
            self.enc_cells[0]=rnn_cell.EmbeddingWrapper(self.enc_cells[0],self.decoder_words,self.embedding_size)
            self.enc_scopes.append("encoder_{}".format(0)) #Word level encoder scope
            self.dec_scopes.append("decoder_{}".format(0)) #Decoder scope
            self.dec_cells.append(self.cell_type(self.cell_size))
            
        create_cell_scopes()

    def create_placeholder(self):
        '''Creating placeholder for encoder inputs,decoder inputs,targets and weights. '''
        #self.encoder1_inputs - A list of 1D batch-sized int32 Tensors, where tensor has batch sized indices for first utterance of dialogue.
        self.encoder1_inputs=[tf.placeholder(tf.int32,[None],name="encoder1_input") for i in range(self.max_len)]
        #self.encoder2_inputs - A list of 1D batch-sized int32 Tensors, where tensor has batch sized indices for second utterance of dialogue.
        self.encoder2_inputs=[tf.placeholder(tf.int32,[None],name="encoder2_input") for i in range(self.max_len)]
        #self.decoder_inputs - A list of 1D batch-sized int32 Tensors, where tensor has batch sized indices for third utterance of dialogue.
        self.decoder_inputs=[tf.placeholder(tf.int32,[None],name="decoder_input") for i in range(self.max_len)]
        #self.concat_dec_ip - Concatenating utterance level output with decoder input ate every time step, where output of utterance level would be self.batch_size*self.cell_size and decoder_input will be of size self.batch_size*self.embedding_size. 
        self.concat_dec_ip=[tf.placeholder(tf.int32,[None,self.cell_size+self.embedding_size],name="concatenated_dec_ip") for i in range(self.max_len)]
        #self.target - a list of integer tensors of batch_size. Each tensor contains the index of true output word.
        self.targets=[tf.placeholder(tf.int32,[None],name="target") for i in range(self.max_len)]
        #self.weights - a list of float tensors(for calculating weighted sequence loss).
        self.weights=[tf.placeholder(tf.float32,[None],name="seq2seq_weights") for i in range(self.max_len)]
        #self.feed_previous - Placeholder considered for making decoder input as previous step decoder output at traning(after 2 epochs) and validation.
        self.feed_previous=tf.placeholder(tf.bool,name='feed_previous')
    
    def hierarchical_encoder(self):
        '''Acts as encoder for word and utterance level.
        Return - enc_states which is final output of utterance level encoder RNN which is of size batch_size*self.cell_size.'''

        enc_states=self.encoder1_inputs+self.encoder2_inputs #Merging self.encoder1_inputs and self.encoder2_inputs list.
        n_steps=self.max_len #Word level encoder steps is same as self.max_len which is 150.
        #Calling self.encoder function, for word level encoder.
        enc_states=self.encoder(enc_states,self.enc_cells[0],self.enc_scopes[0],n_steps)
        #Using bidirectional RNN for utterance level encoder.
        _,utterance_state,_=rnn.bidirectional_rnn(self.enc_cells[1][0],self.enc_cells[1][1],enc_states,dtype=tf.float32)
        return utterance_state

    def encoder(self,inputs,enc_cell,enc_scope,n_steps):
        ''' Performs encoder operation for word and utterance level.
        Args:
            inputs: Combined input list for both the utterances where number of time steps is 2*self.max_len.
            enc_cell: Encoder cell for word level encoder.
            enc_scope: encoder scope for word level encoder.
            n_steps: No of time steps for word level(self.max_len).
        Returns - sentence_states which is a list of final state for word level encoder.'''

        sentence_states=[]
        with tf.variable_scope(enc_scope) as scope:
            for i in range(0,len(inputs),n_steps):
                if i>0:
                    scope.reuse_variables() #Using same rnn cell(parameter sharing) for both encoder inputs.
                _,states=rnn.rnn(enc_cell,inputs[i:i+n_steps],dtype=tf.float32) #Return final state for word level
                sentence_states.append(states) #Two final states for word level encoder.
        return sentence_states
        										
    def hierarchical_decoder(self,utterance_output):
        '''Acts as a decoder.
        Args:
            utterance_output - Final state of utterance level encoder.
        Return - dec_outputs which is list of tensor of size batch_size*self.vocab_size.'''

        dec_outputs=self.decoder(self.decoder_inputs,utterance_output,self.dec_cells[0],self.dec_scopes[0])
        return dec_outputs
  
    def decode(self,concatenated_ip,loop_fn,dec_cell,init_state,utterance_output,dec_scope):
        ''' Performs decoding operation based on loop_fn being empty(at time of training) or loop_fn exists(time of validation)
        Args:
            concatenated_ip: Concatenation of embedding of decoder input words and utterance level encoder output.
            loop_fn: Used differently at time of training and validation. If True(Validation and Testing) takes prevous time step output of    decoder as an input to decoder at next time step. If False(Training) then current time step input is considered. 
            dec_cell: rnn of lstm cell
            utterance_output: Final Output of utterance level encoder.
            dec_scope: Scope for decoder
        Return: 
              outputs: A list of the same length as decoder_inputs of 2D Tensors with
              shape [batch_size x self.cell_size] containing the generated
              outputs.
              state: The state of each decoder cell in each time-step. This is a list
              with length len(decoder_inputs) -- one item for each time-step.
              It is a 2D Tensor of shape [batch_size x self.cell_size].'''

        state = init_state
        outputs = []
        prev = None
        for i, inp in enumerate(concatenated_ip):
            if loop_fn is not None and prev is not None:
                with tf.variable_scope("loop_function", reuse=True):
                    inp = loop_fn(prev, i)
                    inp=tf.concat(1,[utterance_output,inp])
            if i > 0:
                dec_scope.reuse_variables()
            output, state = dec_cell(inp, state)
            outputs.append(output)
            if loop_fn is not None:
                prev = output
        return outputs,state 

    def decoder(self,decoder_inputs,utterance_output,dec_cell,dec_scope):
        ''' Performs decoding operation.
        Args:
            decoder_inputs: Input to decoder at every time step which is a list of length self.max_len having tensor of size as batch_size.
            utterance_output: Final output of utterance level encoder.
            dec_cell: rnn or lstm cell for decoding purpose
            dec_scope: scope for decoder.
        Return - dec_output which is a list of length self.max_len having tensors of size self.batch_size*self.vocab_size.'''

        with tf.variable_scope(dec_scope) as scope:
            init_state=dec_cell.zero_state(self.batch_size,tf.float32)#Initial state for decoder cell.
            max_val = np.sqrt(6. / (self.decoder_words + self.cell_size))
            weights=tf.get_variable("dec_weights",[self.cell_size,self.decoder_words],initializer=tf.random_uniform_initializer(-1.*max_val,max_val))#For projecting decoder output which is of self.batch_size*self.cell_size to self.batch_size*self.vocab_size.
            biases = tf.get_variable("dec_biases",[self.decoder_words],initializer=tf.constant_initializer(0.0))

            def feed_prev_decode(feed_previous_bool):
	        '''Makes two seperate graphs based on feed_previous input given at training and test time.
                Args:
                    feed_previous_bool: Boolean tensor which is True at time of validation and testing and False at time of training.
                Return: dec_output which is a list having tensor of size batch_size*self.cell_size.'''

                dec_embed,loop_fn=seq2seq.get_decoder_embedding(decoder_inputs,self.decoder_words,self.embedding_size,output_projection=(weights,biases),feed_previous=feed_previous_bool)#look for get_decoder_embedding in seq2seq.py 
                concatenated_ip=self.get_dec_concat_ip(dec_embed,utterance_output)
                dec_output,_=self.decode(concatenated_ip,loop_fn,dec_cell,init_state,utterance_output,scope)
                return dec_output
            
            dec_output = control_flow_ops.cond(self.feed_previous,lambda: feed_prev_decode(True),lambda: feed_prev_decode(False)) #Calls feed_prev_decode function based on feed_prev_bool is True or False.
            output_projection=(weights,biases)
            #Projects dec_output from self.batch_size*self.cell_size to self.batch_size*self.vocab_size
            for i in range(len(dec_output)):
                if self.activation!=None:
                    dec_output[i]=self.activation(tf.matmul(dec_output[i],output_projection[0])+output_projection[1])
                else:
                    dec_output[i]=tf.matmul(dec_output[i],output_projection[0])+output_projection[1]
        return dec_output
    
    def get_dec_concat_ip(self,dec_embed,utterance_output):
        '''Performs concatenation of decoder input at every time step with utterance level encoder output. 
        Args:
            dec_embed: A list of length self.max_len having tensor of size batch_size*self.embedding_size.
            utterance_output: output of utterance level encoder.
        Return: self.concat_dec_ip which is a concatenation of embedding of decoder input words and utterance level encoder output.'''

        for(i,inp) in enumerate(dec_embed): #Looping over list having embedding of decoder input words at every time step.    
            self.concat_dec_ip[i]=tf.concat(1,[utterance_output,inp]) #Performing concatenation.
        return self.concat_dec_ip
    
    def inference(self):
        '''Calls encoder and decoder function. 
        Return: logits which is a list(output of decoder) of length self.max_len having tensor of size batch_size*self.vocab_size. '''

        utterance_output=self.hierarchical_encoder()
        logits=self.hierarchical_decoder(utterance_output)
        return logits
 
    def loss(self,logits):
        ''' Calculates sequence loss between logits(decoder output or predicted output) and self.targets(True output).
         Args:
             logits: logits which is a list(output of decoder) of length self.max_len having tensor of size batch_size*self.vocab_size. 
         Return: losses which is loss calculated between predicted output and true output.'''
        
        #self.targets: List of 1D batch-sized int32 Tensors of the same length as logits.
        #self.weights: List of 1D batch-sized float-Tensors of the same length as logits.
        losses=seq2seq.sequence_loss_by_example(logits,self.targets,self.weights)	
        return losses

    def train(self,losses):
        ''' Sets up the training Ops by performing Adam Optimization to minimize loss at every time step. Creates an optimizer and applies the gradients to all trainable variables.
        Args: losses tensor, from seq2seq.sequence_loss().
        Returns:train_op which is Op for training.'''

        parameters=tf.trainable_variables()
        optimizer=tf.train.AdamOptimizer(learning_rate=self.learning_rate,beta1=0.9,beta2=0.999,epsilon=1e-08)
        gradients=tf.gradients(losses,parameters)
        clipped_gradients,norm=tf.clip_by_global_norm(gradients,self.max_gradient_norm)
        global_step=tf.Variable(0,name="global_step",trainable='False')
        #train_op=optimizer.minimize(losses,global_step=global_step)
        train_op=optimizer.apply_gradients(zip(clipped_gradients,parameters),global_step=global_step)
        return train_op
