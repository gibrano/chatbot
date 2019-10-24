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


# Define sampling models
encoder_model = Model(encoder_inputs, encoder_states)

decoder_state_input_h = Input(shape=(latent_dim,))
decoder_state_input_c = Input(shape=(latent_dim,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)
decoder_model = Model( [decoder_inputs] + decoder_states_inputs,[decoder_outputs] + decoder_states)

# Reverse-lookup token index to decode sequences back to
# something readable.
reverse_input_char_index = dict((i, char) for char, i in input_token_index.items())
reverse_target_char_index = dict((i, char) for char, i in target_token_index.items())

def decode_sequence(input_seq):
    # Encode the input as state vectors.
    states_value = encoder_model.predict(input_seq)
    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, 1, num_decoder_tokens))
    # Populate the first character of target sequence with the start character.
    target_seq[0, 0, target_token_index['\t']] = 1.
    # Sampling loop for a batch of sequences
    # (to simplify, here we assume a batch of size 1).
    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        # Sample a token
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = reverse_target_char_index[sampled_token_index]
        decoded_sentence += sampled_char
        # Exit condition: either hit max length
        # or find stop character.
        if (sampled_char == '\n' or len(decoded_sentence) > max_decoder_seq_length):
            stop_condition = True
        # Update the target sequence (of length 1).
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        target_seq[0, 0, sampled_token_index] = 1.
        # Update states
        states_value = [h, c]
    return decoded_sentence


for seq_index in range(10):
    input_seq = encoder_input_data[seq_index: seq_index + 1]
    decoded_sentence = decode_sequence(input_seq)
    print('-----------------------------------------------------')
    print('Input sentence:', input_texts[seq_index])
    print('Decoded sentence:', decoded_sentence)