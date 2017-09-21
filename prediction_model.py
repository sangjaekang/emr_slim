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

def prediction_second_model(emr_rows,emr_cols,emr_depths,num_classes):
    '''
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

    model = Model(inputs=lab_input, outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])

    return model

def prediction_third_model(emr_rows,emr_cols,emr_depths,num_classes):
    lab_input = Input(shape=(emr_rows,emr_cols,emr_depths))
    demo_input = Input(shape=(12,1))
    pres_input = Input(shape=(200,emr_cols,1))

    # 3months(long term predicition)
    model_1 = MaxPooling2D((1,3),strides=(1,2),padding='same')(lab_input)
    model_1 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Activation('relu')(model_1)
    model_1 = Conv2D(4,(1,1), padding='same',kernel_initializer=initializers.he_uniform())(model_1)
    model_1 = BatchNormalization()(model_1)
    model_1 = Activation('relu')(model_1)
    # 2months(middle term predicition)
    model_2 = MaxPooling2D((1,2),strides=(1,2),padding='same')(lab_input)
    model_2 = Conv2D(8,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(model_2)
    model_2 = BatchNormalization()(model_2)
    model_2 = Activation('relu')(model_2)
    model_2 = Conv2D(4,(1,1), padding='same',kernel_initializer=initializers.he_uniform())(model_2)
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
    model_3 = Conv2D(4,(1,1), padding='same',kernel_initializer=initializers.he_uniform())(model_3)
    model_3 = BatchNormalization()(model_3)
    model_3 = Activation('relu')(model_3)
    merged = concatenate([model_1,model_2,model_3],axis=1)
    merged = Flatten()(merged)

    # prescribe CNN model 
    pres_model = Conv2D(8,(200,3), padding='same',kernel_initializer=initializers.he_uniform())(pres_input)
    pres_model = BatchNormalization()(pres_model)
    pres_model = Activation('relu')(pres_model)
    pres_model = MaxPooling2D((1,2),strides=(1,2),padding='same')(pres_model)
    pres_model = Conv2D(4,(emr_rows,3), padding='same',kernel_initializer=initializers.he_uniform())(pres_model)
    pres_model = BatchNormalization()(pres_model)
    pres_model = Activation('relu')(pres_model)
    pres_model = Flatten()(pres_model)

    merged = concatenate([merged,pres_model],axis=1)

    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    f_demo_input = Flatten()(demo_input)
    merged = concatenate([merged,f_demo_input],axis=1)
    merged = Dropout(0.5)(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)
    merged = Dense(100,activation='relu')(merged)
    merged = BatchNormalization()(merged)

    out = Dense(num_classes,activation='softmax')(merged)

    model = Model(inputs=[lab_input,pres_input,demo_input], outputs=out)
    adam = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
    model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
    return model