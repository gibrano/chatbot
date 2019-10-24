import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Dense
import numpy as np

class ChatBotModel:

    def __init__(self):
        self.BATCH_SIZE = 64
        self.EPOCHS = 5000
        self.LATENT_DIM = 256
        self.DATA_PATH = 'data/train.csv'
        self.MODEL_PATH = 'model/s2s_model.h5'

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

    def lstm_model(self):
        tf.debugging.set_log_device_placement(True)
        #strategy = tf.distribute.MirroredStrategy(devices=["/gpu:0", "/gpu:1"])
        strategy = tf.distribute.MirroredStrategy()
        print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
        with strategy.scope():
            encoder_inputs = Input(shape=(None, self.num_encoder_tokens))
            encoder = LSTM(self.LATENT_DIM, return_state=True)
            encoder_outputs, state_h, state_c = encoder(encoder_inputs)
            encoder_states = [state_h, state_c]
            decoder_inputs = Input(shape=(None, self.num_decoder_tokens))
            decoder_lstm = LSTM(self.LATENT_DIM, return_sequences=True, return_state=True)
            decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
            decoder_dense = Dense(self.num_decoder_tokens, activation='softmax')
            decoder_outputs = decoder_dense(decoder_outputs)
            model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
            model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
        return model

    def train(self):
        model.fit([encoder_input_data, decoder_input_data], decoder_target_data, batch_size=batch_size, epochs=epochs, validation_split=0.2)
        # Save model
        model.save(self.MODEL_PATH)