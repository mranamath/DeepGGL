#!/usr/bin/env python

'''
Introduction:
    -Train DeepGGL
Author:
    Masud Rana (mrana10@kennesaw.edu)
Last Updated:
    Sep 9, 2025

'''

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, Flatten, Dense,
    Attention, BatchNormalization,
    Dropout, Activation, Add
)

from tensorflow.keras.models import Model
from tensorflow.keras.regularizers import l2

from sklearn.preprocessing import StandardScaler
import joblib

import tensorflow_probability as tfp

import argparse

print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

def PCC(y_true, y_pred):
    return tfp.stats.correlation(y_true, y_pred, sample_axis=0)

def Weighted_PCC_RMSE(y_true, y_pred):
    alpha=0.7
    rmse = tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))
    pcc = PCC(y_true, y_pred)
    return alpha * (1 - pcc) + (1 - alpha) * rmse


def residual_block(input_tensor, filters, kernel_size=5, strides=1):
    shortcut = input_tensor
   
    # First convolution
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second convolution
    x = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding='same')(x)
    x = BatchNormalization()(x)

    # Add input tensor (skip connection)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)

    return x

def create_model(input_shape):
    # Define the input layers
    input_1 = Input(shape=input_shape)

    # Define CNN layers with residual blocks
    def create_cnn_layers(input_layer, kernel_size=7):
        x = Conv2D(32, kernel_size=kernel_size, activation='relu')(input_layer)

        x = Conv2D(64, kernel_size=kernel_size, padding='same')(x)  
        x = residual_block(x, 64, kernel_size=kernel_size)  

        x = Conv2D(128, kernel_size=kernel_size, padding='same')(x)  
        x = residual_block(x, 128, kernel_size=kernel_size) 

        x = Flatten()(x)
        return x

    # Create CNN layers for each input
    output_1 = create_cnn_layers(input_1)


    # Add self attention mechanism
    attention = Attention()([output_1, output_1])

    # Add fully connected layers with batch normalization, dropout, and L2 regularization
    fc1 = Dense(128, activation='relu', kernel_regularizer=l2(0.01))(attention)
    fc1 = BatchNormalization()(fc1)
    fc1 = Dropout(0.1)(fc1)

    fc2 = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(fc1)
    fc2 = BatchNormalization()(fc2)
    fc2 = Dropout(0.1)(fc2)

    # Add the output layer (for regression)
    output = Dense(1, kernel_regularizer=l2(0.01))(fc2)

    # Define the model
    model = Model(inputs=[input_1], outputs=output)
    
    return model


def myprint(s):
    with open('./model_summary.txt','a') as f:
        print(s, file=f)

if __name__=="__main__":

    parser = argparse.ArgumentParser(description="CNN with Residual Connection and Attention",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)                    

    parser.add_argument('-tf', '--train_feature', help='file path of training feature')
    parser.add_argument('-vf', '--val_feature', help='file path of validation feature')  
    parser.add_argument('-m','--model_name', type=str, help='model name')

    args = parser.parse_args()

    df_train = pd.read_csv(args.train_feature)

    df_val = pd.read_csv(args.val_feature)

    ytrain = df_train['pK'].values
    yval = df_val['pK'].values

    Xtrn = df_train.drop(['PDBID','pK'], axis=1)
    Xtrn = Xtrn.values

    Xval = df_val.drop(['PDBID','pK'], axis=1)
    Xval = Xval.values

    print('Shape of datasets')
    print('Xtrain: ', Xtrn.shape)
    print('Xval: ', Xval.shape)

    train_scaler = StandardScaler()
    train_scaler.fit(Xtrn)
    joblib.dump(train_scaler, 'training_scaler.scaler')


    # Example shapes for feature matrices
    input_shape = (74, 112, 3)

    Xtrn = train_scaler.transform(Xtrn).reshape(-1,input_shape[0],input_shape[1],input_shape[2])

    Xval = train_scaler.transform(Xval).reshape(-1,input_shape[0],input_shape[1],input_shape[2])

    print("DataSet Scaled")

    print('Shape of reshaped datasets')
    print('Xtrain: ', Xtrn.shape)
    print('Xval: ', Xval.shape)

    print('ytrain: ', ytrain.shape)
    print('yval: ', yval.shape)

    del df_train, df_val

    # Create the model
    model = create_model(input_shape)

    # Print the model summary
    model.summary(print_fn=myprint)

    # Compile the model
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, decay=1e-6, clipvalue=0.01)
    model.compile(optimizer=sgd, 
                  loss= Weighted_PCC_RMSE,
                    metrics=['mse'])

    # Callback
    stop = tf.keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.001, patience=20,
                                        verbose=1, mode='auto')
    logger = tf.keras.callbacks.CSVLogger(f'{args.model_name}.log', separator=',', append=False)

    model_outname=f'{args.model_name}.h5'
    bestmodel = tf.keras.callbacks.ModelCheckpoint(filepath=model_outname, verbose=1, save_best_only=True)
    
    callbacks = [stop, logger, bestmodel]

    history = model.fit([Xtrn], ytrain, 
                        validation_data=([Xval], yval),   
                    epochs = 300,
                    batch_size = 64,
                    verbose=1,
                    callbacks=callbacks)

