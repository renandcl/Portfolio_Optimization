import numpy as np

from keras.layers import Input
from keras.layers import Reshape
from keras.layers import BatchNormalization
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import MultiHeadAttention
from keras.layers import AdditiveAttention
from keras.layers import Concatenate
from keras.layers import Dense
from keras.layers import GlobalAveragePooling1D
from keras.layers import Dot
from keras.optimizers import Adam
from keras.models import Model
from keras import regularizers
from keras.utils import plot_model
import keras_tuner
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split



# Load the TensorBoard notebook extension
# %load_ext tensorboard
import tensorflow as tf


def LSTM_n_AddAtt(l2_1, l2_2, lr, tickers, timesteps, portfolios):

    hidden_size = 32
    
    # Inputs
    inputs = Input(shape=(timesteps, tickers), name='inputs')

    # Batch Normalization
    batch_1 = BatchNormalization(name='batchnorm_1')(inputs)

    # Encoder LSTM
    lstm = LSTM(hidden_size, activation='sigmoid', return_state=True, return_sequences=True, kernel_regularizer= l2_1, name='lstm')
    all_state_h, final_state_h, state_c = lstm(batch_1)

    # Batch Normalization
    batch_lstm = BatchNormalization(name='batchnorm_lstm')(final_state_h)

    # Reshape the encoder LSTM output
    reshape_1 = Reshape((batch_lstm.shape[-1],1), name='reshape_1')(batch_lstm)

    # Additive Attention layer
    add_attn_layer = AdditiveAttention(name='addattention_layer')
    attn_outs,_ = add_attn_layer([batch_1,reshape_1],  return_attention_scores = True)

    # Concat attention output and decoder LSTM output
    concat_input = Concatenate(axis=-1, name='concat_layer')([batch_1, attn_outs])

    # # Dense layer
    dense_1 = Dense(hidden_size,activation='tanh',kernel_regularizer= l2_2, name='dense_1')
    dense_concat = dense_1(concat_input)

    # Global Average Pooling
    avg_pool = GlobalAveragePooling1D(name='avg_pool')(dense_concat)

    # Batch Normalization
    batch_2 = BatchNormalization(name='batchnorm_2')(avg_pool)

    # Softmax layer
    pred = Dense(portfolios, activation='softmax',kernel_regularizer= l2_2, name='output')(batch_2)
    
    # Model
    model = Model(inputs=[inputs], outputs=[pred])

    # Optimizer
    optimizer = Adam(learning_rate=lr)

    # Compile model    
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics='accuracy')

    return model


def GRU_n_AddAtt(l2_1, l2_2, lr, tickers, timesteps, portfolios):

    hidden_size = 32
    
    # Inputs
    inputs = Input(shape=(timesteps, tickers), name='inputs')

    # Batch Normalization
    batch_1 = BatchNormalization(name='batchnorm_1')(inputs)

    # Encoder GRU
    gru = GRU(hidden_size, activation='sigmoid', return_state=True, return_sequences=True, kernel_regularizer= l2_1, name='gru')
    all_state_h, final_state_h = gru(batch_1)

    # Batch Normalization
    batch_gru = BatchNormalization(name='batchnorm_gru')(final_state_h)

    # Reshape the encoder GRU output
    reshape_1 = Reshape((batch_gru.shape[-1],1), name='reshape_1')(batch_gru)

    # Additive Attention layer
    add_attn_layer = AdditiveAttention(name='addattention_layer')
    attn_outs,_ = add_attn_layer([batch_1,reshape_1],  return_attention_scores = True)

    # Concat attention output and decoder GRU output
    concat_input = Concatenate(axis=-1, name='concat_layer')([batch_1, attn_outs])

    # # Dense layer
    dense_1 = Dense(hidden_size,activation='tanh',kernel_regularizer= l2_2, name='dense_1')
    dense_concat = dense_1(concat_input)

    # Global Average Pooling
    avg_pool = GlobalAveragePooling1D(name='avg_pool')(dense_concat)

    # Batch Normalization
    batch_2 = BatchNormalization(name='batchnorm_2')(avg_pool)

    # Softmax layer
    pred = Dense(portfolios, activation='softmax',kernel_regularizer= l2_2, name='output')(batch_2)
    
    # Model
    model = Model(inputs=[inputs], outputs=[pred])

    # Optimizer
    optimizer = Adam(learning_rate=lr)

    # Compile model    
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics='accuracy')

    return model


def LSTM_n_SelfAtt(l2_1, l2_2, lr, tickers, timesteps, portfolios):

    hidden_size = 32
    
    # Inputs
    inputs = Input(shape=(timesteps, tickers), name='inputs')

    # Batch Normalization
    batch_1 = BatchNormalization(name='batchnorm_1')(inputs)
    
    # Self Attention layer
    self_attn_layer = MultiHeadAttention(num_heads=1, key_dim=tickers, name='self_attention_layer')
    out_tensor, weights = self_attn_layer(batch_1, batch_1, return_attention_scores = True)

    # # Dot product layer
    dot_product = Dot(axes=(2,1), name='dot_product')([weights,batch_1])
    
    # reshape dot product to (batchsize, timesteps, tickers)
    dot_product = Reshape((timesteps, tickers))(dot_product)
    
    # # # Encoder LSTM
    lstm = LSTM(hidden_size, activation='sigmoid', return_state=True, return_sequences=True, kernel_regularizer= l2_1, name='lstm')
    all_state_h, final_state_h, state_c = lstm(dot_product)

    # Batch Normalization
    batch_lstm = BatchNormalization(name='batchnorm_lstm')(final_state_h)

    # Softmax layer
    pred = Dense(portfolios, activation='softmax',kernel_regularizer= l2_2, name='output')(batch_lstm)
    
    # Model
    model = Model(inputs=[inputs], outputs=[pred])

    # Optimizer
    optimizer = Adam(learning_rate=lr)

    # Compile model    
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics='accuracy')

    return model


def GRU_n_SelfAtt(l2_1, l2_2, lr, tickers, timesteps, portfolios):

    hidden_size = 32
    
    # Inputs
    inputs = Input(shape=(timesteps, tickers), name='inputs')

    # Batch Normalization
    batch_1 = BatchNormalization(name='batchnorm_1')(inputs)
    
    # Self Attention layer
    self_attn_layer = MultiHeadAttention(num_heads=1, key_dim=tickers, name='self_attention_layer')
    out_tensor, weights = self_attn_layer(batch_1, batch_1, return_attention_scores = True)

    # # Dot product layer
    dot_product = Dot(axes=(2,1), name='dot_product')([weights,batch_1])
    
    # reshape dot product to (batchsize, timesteps, tickers)
    dot_product = Reshape((timesteps, tickers))(dot_product)
    
    # # # Encoder GRU
    gru = GRU(hidden_size, activation='sigmoid', return_state=True, return_sequences=True, kernel_regularizer= l2_1, name='gru')
    all_state_h, final_state_h = gru(dot_product)

    # Batch Normalization
    batch_gru = BatchNormalization(name='batchnorm_gru')(final_state_h)

    # Softmax layer
    pred = Dense(portfolios, activation='softmax',kernel_regularizer= l2_2, name='output')(batch_gru)
    
    # Model
    model = Model(inputs=[inputs], outputs=[pred])

    # Optimizer
    optimizer = Adam(learning_rate=lr)

    # Compile model    
    model.compile(optimizer=optimizer, loss= 'categorical_crossentropy', metrics='accuracy')

    return model


def gap_kfold(x,y,kfold,test_position,gap_before,gap_after):
    fold_size = int((x.shape[0]) / kfold)
    x_train = []
    y_train = []
    x_test = []
    y_test = []
    gap_positions = list(range(0 if test_position-gap_before < 0 else test_position-gap_before, kfold if test_position+gap_after > kfold else test_position+gap_after+1))
    gap_positions.pop(gap_positions.index(test_position))
    for i in range(kfold):
        if i == test_position:
            x_test.append(x[i*fold_size:i*fold_size+fold_size,:,:])
            y_test.append(y[i*fold_size:i*fold_size+fold_size,:])
        elif i in gap_positions:
            continue
        else:
            x_train.append(x[i*fold_size:i*fold_size+fold_size,:,:])
            y_train.append(y[i*fold_size:i*fold_size+fold_size,:])
    x_train = np.concatenate(x_train)
    y_train = np.concatenate(y_train)
    x_test = np.concatenate(x_test)
    y_test = np.concatenate(y_test)
    return x_train, y_train, x_test, y_test