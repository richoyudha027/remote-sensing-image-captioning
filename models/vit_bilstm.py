import json
import numpy as np


VIT_BILSTM_WEIGHT_PATH     = './weights/vit_bilstm_best_model.h5'
VIT_BILSTM_TOKENIZER_PATH = './weights/vit_bilstm_tokenizer.json'
VIT_BILSTM_CONFIG_PATH    = './weights/vit_bilstm_config.json'


def load_vit_bilstm():
    print('[1/3] Loading ViT-BiLSTM.')
    import tensorflow as tf
    import tensorflow_hub as hub
    from tensorflow.keras.preprocessing.text import tokenizer_from_json
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from tensorflow.keras.layers import (Input, Dropout, Dense, Embedding, LSTM,
                                          Reshape, concatenate, Bidirectional,
                                          LayerNormalization, Flatten, add)
    from tensorflow.keras.models import Model

    with open(VIT_BILSTM_TOKENIZER_PATH, 'r') as f:
        tokenizer = tokenizer_from_json(f.read())
    with open(VIT_BILSTM_CONFIG_PATH, 'r') as f:
        config = json.load(f)
    max_length = config['max_length']
    vocab_size = config['vocab_size']

    vit_url = 'https://tfhub.dev/sayakpaul/vit_b16_fe/1'
    class ViTFeatureExtractor(tf.keras.Model):
        def __init__(self):
            super().__init__()
            self.vit = hub.KerasLayer(vit_url, trainable=False)
        def call(self, images):
            return self.vit((images * 2.0) - 1.0)

    feature_extractor = ViTFeatureExtractor()

    input1 = Input(shape=(768,))
    input2 = Input(shape=(max_length,))

    img_features = Dense(512, activation='relu')(input1)
    img_features = Dropout(0.3)(img_features)
    img_features = Dense(512, activation='relu')(img_features)
    img_features_reshaped = Reshape((1, 512))(img_features)

    sentence_features = Embedding(vocab_size, 512, mask_zero=False)(input2)
    merged = concatenate([img_features_reshaped, sentence_features], axis=1)

    bi_lstm_out = Bidirectional(LSTM(512, return_sequences=True, dropout=0.2, recurrent_dropout=0.2))(merged)
    bi_lstm_out = LayerNormalization()(bi_lstm_out)
    bi_lstm_out2 = Bidirectional(LSTM(512, dropout=0.2, recurrent_dropout=0.2))(bi_lstm_out)
    bi_lstm_out2 = LayerNormalization()(bi_lstm_out2)

    X = Dense(512, activation='relu')(bi_lstm_out2)
    X = Dropout(0.4)(X)

    img_features_flat = Flatten()(img_features_reshaped)
    X = add([X, img_features_flat])

    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    output = Dense(vocab_size, activation='softmax')(X)

    caption_model = Model(inputs=[input1, input2], outputs=output)
    caption_model.load_weights(VIT_BILSTM_MODEL_PATH)

    return caption_model, feature_extractor, tokenizer, max_length, pad_sequences


def predict_vit_bilstm(image_path, caption_model, feature_extractor, tokenizer, max_length, pad_sequences):
    import tensorflow as tf

    img = tf.io.read_file(image_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))
    img = tf.cast(img, tf.float32) / 255.0
    img = (img * 2.0) - 1.0
    img = tf.expand_dims(img, 0)
    img_feature = feature_extractor.predict(img, verbose=0)

    in_text = 'startseq'
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        yhat = caption_model.predict([img_feature, sequence], verbose=0)
        yhat = np.argmax(yhat)
        word = None
        for w, idx in tokenizer.word_index.items():
            if idx == yhat:
                word = w
                break
        if word is None or word == 'endseq':
            break
        in_text += ' ' + word

    final_caption = in_text.replace('startseq', '').strip()
    if final_caption and final_caption[-1] not in '.!?':
        final_caption += '.'
    return final_caption