import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Input
import numpy as np

class ChatBot:
    
    def __init__(self):
        self.BATCH_SIZE = 64
        self.EPOCHS = 5000
        self.LATENT_DIM = 256
        self.DATA_PATH = 'data/train.csv'
        self.MODEL_PATH = 'model/s2s_model.h5'
        self.load_data()
        self.tokenize()
        self.load_lstm_model()

    def load_data(self): 
        self.input_texts = []
        self.target_texts = []
        self.input_characters = set()
        self.target_characters = set()
        with open(self.DATA_PATH, 'r', encoding='utf-8') as f:
            lines = f.read().split('\n')

        for line in lines:
            input_text, target_text = line.split('\t')
            target_text = '\t' + target_text + '\n'
            self.input_texts.append(input_text)
            self.target_texts.append(target_text)
            for char in input_text:
                if char not in self.input_characters:
                    self.input_characters.add(char)
            for char in target_text:
                if char not in self.target_characters:
                    self.target_characters.add(char)

    def tokenize(self):
        input_characters = sorted(list(self.input_characters))
        target_characters = sorted(list(self.target_characters))
        self.num_encoder_tokens = len(input_characters)
        self.num_decoder_tokens = len(target_characters)
        self.max_encoder_seq_length = max([len(txt) for txt in self.input_texts])
        self.max_decoder_seq_length = max([len(txt) for txt in self.target_texts])

        self.input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
        self.target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

        self.encoder_input_data = np.zeros((len(self.input_texts), self.max_encoder_seq_length, self.num_encoder_tokens),dtype='float32')
        self.decoder_input_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),dtype='float32')
        self.decoder_target_data = np.zeros((len(self.input_texts), self.max_decoder_seq_length, self.num_decoder_tokens),dtype='float32')

        for i, (input_text, target_text) in enumerate(zip(self.input_texts, self.target_texts)):
            for t, char in enumerate(input_text):
                self.encoder_input_data[i, t, self.input_token_index[char]] = 1.
            for t, char in enumerate(target_text):
                self.decoder_input_data[i, t, self.target_token_index[char]] = 1.
                if t > 0:
                    self.decoder_target_data[i, t - 1, self.target_token_index[char]] = 1.

        self.reverse_input_char_index = dict((i, char) for char, i in self.input_token_index.items())
        self.reverse_target_char_index = dict((i, char) for char, i in self.target_token_index.items())            

    def load_lstm_model(self):
        model = load_model(self.MODEL_PATH)
        encoder_inputs = model.input[0]
        encoder_outputs, state_h_enc, state_c_enc = model.layers[2].output 
        encoder_states = [state_h_enc, state_c_enc]
        self.encoder_model = Model(encoder_inputs, encoder_states)

        decoder_inputs = model.input[1] 
        decoder_state_input_h = Input(shape=(self.LATENT_DIM,), name='input_3')
        decoder_state_input_c = Input(shape=(self.LATENT_DIM,), name='input_4')
        decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
        decoder_lstm = model.layers[3]
        decoder_outputs, state_h_dec, state_c_dec = decoder_lstm( decoder_inputs, initial_state=decoder_states_inputs)
        decoder_states = [state_h_dec, state_c_dec]
        decoder_dense = model.layers[4]
        decoder_outputs = decoder_dense(decoder_outputs)
        self.decoder_model = Model( [decoder_inputs] + decoder_states_inputs,  [decoder_outputs] + decoder_states)

    def decode_sequence(self, input_seq):
        states_value = self.encoder_model.predict(input_seq)
        target_seq = np.zeros((1, 1, self.num_decoder_tokens))
        target_seq[0, 0, self.target_token_index['\t']] = 1.
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = self.decoder_model.predict([target_seq] + states_value)
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = self.reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char
            if (sampled_char == '\n' or len(decoded_sentence) > self.max_decoder_seq_length):
                stop_condition = True
            target_seq = np.zeros((1, 1, self.num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.
            states_value = [h, c]
        return decoded_sentence

    def run(self, input_text):
        input_texts2 = [input_text]
        encoder_input_data = np.zeros((1, self.max_encoder_seq_length, self.num_encoder_tokens), dtype='float32')
        for i, input_text in enumerate(input_texts2):
            for t, char in enumerate(input_text):
                encoder_input_data[i, t, self.input_token_index[char]] = 1.

        input_seq = encoder_input_data[0: 1]
        decoded_sentence = self.decode_sequence(input_seq)
        return decoded_sentence