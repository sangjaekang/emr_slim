# -*- coding: utf-8 -*-
import sys, os, re
import time
import random

os_path = os.path.abspath('./') ; find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

import keras
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation,Flatten
from keras.layers.pooling import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization
from keras.layers import concatenate
from keras.models import Model

from keras.optimizers import Adam
from keras.utils import np_utils
from keras import initializers
#from keras_tqdm import TQDMNotebookCallback
from keras.callbacks import EarlyStopping, ModelCheckpoint
from impute_mean import *

import matplotlib
matplotlib.use('Agg')   
import matplotlib.pyplot as plt

def prediction_first_model(emr_rows,emr_cols,emr_depths,num_classes):
    '''
    model input
        1. labtest 중 평균값 matrix
        2. boolean matrix
    model output
        1. normal - hyper - hypo
    '''
    lab_input = Input(shape=(emr_rows,emr_cols,emr_depths))
    # 3months(long term predicition)
    model_1 = MaxPooling2D((1,3),strides=(1,2),padding='same')(lab_input)
    model_1 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Activation('relu')(model_1)
    # 2months(middle term predicition)
    model_2 = MaxPooling2D((1,2),strides=(1,2),padding='same')(lab_input)
    model_2 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Activation('relu')(model_2)
    # 1months(short term predicition)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(lab_input)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)
    model_3 = MaxPooling2D((1,2),strides=(1,2),padding='same')(model_3)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_3)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)

    merged = concatenate([model_1,model_2,model_3],axis=1)
    merged = Flatten()(merged)

    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)

    out = Dense(num_classes,activation='softmax')(merged)

    model = Model(inputs=lab_input, outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def prediction_MRCNN(emr_rows,emr_cols,emr_depths,num_classes):
    '''
    MRCNN : Multi-Resolution Convolutional Neural Network
    model input
        1. labtest 중 평균값 matrix
        2. boolean matrix
        3. age&sex factor
    model output
        1. normal - hyper - hypo
    '''
    lab_input = Input(shape=(emr_rows,emr_cols,emr_depths))
    demo_input = Input(shape=(12,1))
    # 3months(long term predicition)
    model_1 = MaxPooling2D((1,3),strides=(1,2),padding='same')(lab_input)
    model_1 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Activation('relu')(model_1)
    # 2months(middle term predicition)
    model_2 = MaxPooling2D((1,2),strides=(1,2),padding='same')(lab_input)
    model_2 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Activation('relu')(model_2)
    # 1months(short term predicition)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(lab_input)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)
    model_3 = MaxPooling2D((1,2),strides=(1,2),padding='same')(model_3)
    model_3 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_3)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)

    merged = concatenate([model_1,model_2,model_3],axis=1)
    merged = Flatten()(merged)

    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)
    f_demo_input = Flatten()(demo_input) # make democratic input flatten
    merged = concatenate([merged,f_demo_input],axis=1)
    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)

    out = Dense(num_classes,activation='softmax')(merged)

    model = Model(inputs=[lab_input,demo_input], outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    return model


def prediction_LABCOMBCNN(emr_rows,emr_cols,emr_depths,num_classes):
    lab_input = Input(shape=(emr_rows,emr_cols,emr_depths))
    demo_input = Input(shape=(12,1))

    vertical_conv_model = Conv2D(emr_rows,(emr_rows,1), padding='same',kernel_initializer=initializers.he_uniform())(lab_input)
    vertical_conv_model = BatchNormalization()(vertical_conv_model)
    vertical_conv_model = Activation('relu')(vertical_conv_model)
    vertical_conv_model = Conv2D(emr_rows,(emr_rows,1), padding='same',kernel_initializer=initializers.he_uniform())(vertical_conv_model)
    vertical_conv_model = BatchNormalization()(vertical_conv_model)
    vertical_conv_model = Activation('relu')(vertical_conv_model)
    
    max_pool_model   = MaxPooling2D((1,3),strides=(1,2),padding='same')(vertical_conv_model)
    temp_conv_model =  Conv2D(64,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(max_pool_model)
    temp_conv_model = BatchNormalization()(temp_conv_model)
    temp_conv_model = Activation('relu')(temp_conv_model)

    fc_model = Flatten()(temp_conv_model)
    fc_model = Dropout(0.5)(fc_model)
    fc_model = Dense(100,activation='relu')(fc_model)
    f_demo_input = Flatten()(demo_input) # make democratic input flatten
    fc_model = concatenate([fc_model,f_demo_input],axis=1)
    fc_model = Dropout(0.5)(fc_model)
    fc_model = Dense(100,activation='relu')(fc_model)
    fc_model = BatchNormalization()(fc_model)

    out = Dense(num_classes, activation='softmax')(fc_model)

    model = Model(inputs=[lab_input,demo_input],outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def prediction_LSTM(emr_rows,emr_cols,emr_depths,num_classes):
    lab_input_1 = Input(shape=(emr_rows,emr_cols))
    lab_input_2 = Input(shape=(emr_rows,emr_cols))
    lab_input_3 = Input(shape=(emr_rows,emr_cols))
    
    LSTM_1 = LSTM(500)(lab_input_1)
    LSTM_2 = LSTM(500)(lab_input_2)
    LSTM_3 = LSTM(500)(lab_input_3)
    LSTM_concat = concatenate([LSTM_1,LSTM_2,LSTM_3],axis=1)
    fc_model = Dropout(0.5)(LSTM_concat)
    fc_model = Dense(100,activation='relu')(fc_model)
    fc_model = Dropout(0.5)(LSTM_concat)
    fc_model = Dense(100,activation='relu')(fc_model)

    out = Dense(num_classes,activation='softmax')(fc_model)
    
    model = Model(inputs=[lab_input_1,lab_input_2,lab_input_3], outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    return model
    