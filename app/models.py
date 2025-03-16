import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Input, LSTM, GRU, Dense, Dropout

class AttentionLayer(Layer):
    """
    Кастомный слой внимания для временных рядов.
    """
    def __init__(self, **kwargs):
        super(AttentionLayer, self).__init__(**kwargs)
    
    def build(self, input_shape):
        hidden_dim = input_shape[-1]
        self.W = self.add_weight(
            name='att_weight',
            shape=(hidden_dim, 1),
            initializer='glorot_uniform'
        )
        self.b = self.add_weight(
            name='att_bias',
            shape=(input_shape[1], 1),
            initializer='zeros'
        )
        super(AttentionLayer, self).build(input_shape)
    
    def call(self, inputs):
        e = tf.tensordot(inputs, self.W, axes=[[2], [0]]) + self.b
        e = tf.squeeze(e, axis=-1)
        alpha = tf.nn.softmax(e)
        alpha = tf.expand_dims(alpha, axis=-1)
        context = inputs * alpha
        context = tf.reduce_sum(context, axis=1)
        return context

def build_model(seq_length, num_features, lstm_units, gru_units, dense_units, dropout_rate, learning_rate):
    """
    Создаёт и компилирует модель на основе входных гиперпараметров.
    
    Parameters:
        seq_length (int): Длина входной последовательности.
        num_features (int): Число признаков.
        lstm_units (int): Число единиц в слое LSTM.
        gru_units (int): Число единиц в слое GRU.
        dense_units (int): Число единиц в Dense-слое.
        dropout_rate (float): Доля dropout.
        learning_rate (float): Скорость обучения.
        
    Returns:
        compiled model (tf.keras.Model)
    """
    inp = Input(shape=(seq_length, num_features))
    x = LSTM(lstm_units, return_sequences=True)(inp)
    x = Dropout(dropout_rate)(x)
    x = GRU(gru_units, return_sequences=True)(x)
    x = Dropout(dropout_rate)(x)
    x = AttentionLayer()(x)
    x = Dense(dense_units, activation="relu")(x)
    x = Dropout(dropout_rate)(x)
    out = Dense(1)(x)
    
    model = Model(inputs=inp, outputs=out)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss="mse")
    return model
