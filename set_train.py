# -*- coding: utf-8 -*-
'''
보다 빠른 속도로 학습을 시키기 위해. shared array 방식을 이용
2개의 shared array를 두어서,
각 array를 번갈아 가며 'dataset 만들기'와 'dataset 학습'을 함
이는 keras method인 fit_generator(multiprocessing 내장된 함수)를 이용하는 것보다 월등히 학습 속도가 빨랐음．
학습할 때 약간씩 오버피팅을 시킴으로써, 최대한 학습을 이끌어내는 방식으로 속도를 높임
(지나치지 않게 earlystoppng callback을 두긴 함)
'''
import sys, os, re
import time
import random
import argparse

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

from keras.optimizers import SGD
from keras.optimizers import Adam
from keras.utils import np_utils
from keras import initializers

from keras.callbacks import EarlyStopping, ModelCheckpoint

from multiprocessing import Pool,Queue,Lock,Array,Process
import ctypes

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from construct_demo import *
from impute_mean import *
from prediction_model import *

def get_np(input_dir,shuffling=True):
    global num_classes
    output_num = random.choice(range(0,num_classes))
    folder_path = input_dir+'/{}/'.format(output_num)
    file_name = random.choice(os.listdir(folder_path))
    file_path = folder_path+file_name
    return get_np_array_emr(file_path,shuffling=shuffling), get_np_array_demo(file_name.split('_')[0]) ,int(output_num)

def batch_maker(input_dir, batch_size, shuffling=True):
    global num_classes
    while 1:
        pool = Pool(processes=8)
        results = [pool.apply_async(get_np,(input_dir,shuffling,)) for _ in xrange(batch_size)]
        feature_list=[]; demo_list=[]; label_list=[]

        for result in results:
            feature, demo, label = result.get()
            feature_list.append(feature)
            demo_list.append(demo)
            label_list.append(label)
        pool.close()
        pool.join()
        batch_features = np.stack(feature_list,axis=0)
        batch_demos = np.stack(demo_list,axis=0).reshape(batch_size,12,1)
        batch_labels = np_utils.to_categorical(label_list,num_classes)    
        
        return batch_features, batch_demos, batch_labels

def label_generator(input_dir,label_num=None,batch_size=None):
    label_dir = input_dir + '/{}/'
    if batch_size is None:
        if label_num is None:
            batch_size = len(os.listdir(label_dir.format(2)))
        else:
            batch_size = len(os.listdir(label_dir.format(label_num)))
    count = batch_size
    
    if label_num is None:
        label_path = label_dir.format(random.choice(range(0,3)))
    else : 
        label_path = label_dir.format(label_num)
        if count >= len(os.listdir(label_path)):
            count = len(os.listdir(label_path))-1

    feature_list = []; demo_list = []; label_list = []
    while count>0:
        feature_path = os.listdir(label_path)[count]
        feature = get_np_array_emr(label_path+feature_path,shuffling=False)
        demo = get_np_array_demo(feature_path.split('_')[0])
        label_list.append(label_num)
        feature_list.append(feature)
        demo_list.append(demo)
        count=count-1 

    batch_features = np.stack(feature_list,axis=0)
    batch_demos = np.stack(demo_list,axis=0).reshape(batch_size,12,1)
    batch_labels = np_utils.to_categorical(label_list,3)
    return batch_features, batch_demos, batch_labels

def train_generator_ml(dataset_size,n_calculation,
                                     arr_ft_1,arr_demo_1,arr_la_1,
                                     arr_ft_2,arr_demo_2,arr_la_2,
                                     input_dir=None):    
    turn_flag = 1
    train_dir = input_dir + '/train/'
    while n_calculation>0:    
        if turn_flag is 1 : 
            with arr_ft_1.get_lock():
                with arr_demo_1.get_lock():
                    with arr_la_1.get_lock():
                        start_time = time.time()
                        print("train get arr_ft_1 lock")
                        x_train,demo_train, y_train = batch_maker(train_dir,dataset_size)
                        nparr_ft_1 = np.frombuffer(arr_ft_1.get_obj())
                        nparr_ft_1[:] = x_train.flatten()
                        nparr_demo_1 = np.frombuffer(arr_demo_1.get_obj())
                        nparr_demo_1[:] = demo_train.flatten()
                        nparr_la_1 = np.frombuffer(arr_la_1.get_obj())
                        nparr_la_1[:] = y_train.flatten()
                        turn_flag = 2
                        print("train unlock arr_ft_1 --- time consumed : {}".format(time.time()-start_time))
        else :
            with arr_ft_2.get_lock():
                with arr_demo_2.get_lock():
                    with arr_la_2.get_lock():
                        start_time = time.time()
                        print("train get arr_ft_2 lock")
                        x_train,demo_train, y_train = batch_maker(train_dir,dataset_size)
                        nparr_ft_2 = np.frombuffer(arr_ft_2.get_obj())
                        nparr_ft_2[:] = x_train.flatten()
                        nparr_demo_2 = np.frombuffer(arr_demo_2.get_obj())
                        nparr_demo_2[:] = demo_train.flatten()
                        nparr_la_2 = np.frombuffer(arr_la_2.get_obj())
                        nparr_la_2[:] = y_train.flatten()
                        turn_flag = 1
                        print("train unlock arr_ft_2 --- time consumed : {}".format(time.time()-start_time))
        time.sleep(1)
        n_calculation = n_calculation -1
        
def fit_train_ml(model,data_shape,
                 arr_ft_1,arr_demo_1,arr_la_1,arr_ft_2,arr_demo_2,arr_la_2,
                 n_calculation,validation_split=0.33,batch_size=256,epochs=200,input_dir=None):
    dataset_size,n_rows,n_cols,n_depths,n_labels = data_shape
    turn_flag = 1
    # save the model_parameter
    if not os.path.isdir(input_dir+'/model_save'):
        os.makedirs(input_dir+'/model_save')
    file_name = input_dir + '/model_save/{val_acc:.3f}.hdf5'
    checkpoint = ModelCheckpoint(file_name, monitor='val_acc', verbose=0, save_best_only=True, mode='max')
    earlystopping = EarlyStopping(monitor='val_loss',min_delta=0, patience=10,verbose=0,mode='auto')
    callback_list = [checkpoint,earlystopping]

    acc_list = []; val_acc_list = []
    loss_list = []; val_loss_list = []

    while n_calculation > 0:
        if turn_flag is 1:
            with arr_ft_1.get_lock():
                with arr_demo_1.get_lock():
                    with arr_la_1.get_lock():
                        start_time = time.time()
                        print("fit_train_ml start!")
                        sys.stdout.flush() 
                        nparr_ft_1 = np.frombuffer(arr_ft_1.get_obj())
                        nparr_ft_1 = np.reshape(nparr_ft_1,(dataset_size,n_rows,n_cols,n_depths)) 
                        nparr_demo_1 = np.frombuffer(arr_demo_1.get_obj())
                        nparr_demo_1 = np.reshape(nparr_demo_1,(dataset_size,12,1)) 
                        nparr_la_1 = np.frombuffer(arr_la_1.get_obj())
                        nparr_la_1 = np.reshape(nparr_la_1,(dataset_size,n_labels))
                        history = model.fit([nparr_ft_1,nparr_demo_1], nparr_la_1,
                                            validation_split=validation_split, batch_size=batch_size,
                                            epochs=epochs,callbacks=callback_list,verbose=0)
                        print("fit_train_ml end -- {}".format(time.time()-start_time))
                        sys.stdout.flush()
                        acc_list.extend(history.history['acc'])
                        val_acc_list.extend(history.history['val_acc'])
                        loss_list.extend(history.history['loss'])
                        val_loss_list.extend(history.history['val_loss'])
                        turn_flag = 2
        else :
            with arr_ft_2.get_lock():
                with arr_demo_2.get_lock():
                    with arr_la_2.get_lock():
                        start_time = time.time()
                        print("fit_train_ml start!")
                        sys.stdout.flush()
                        nparr_ft_2 = np.frombuffer(arr_ft_2.get_obj())
                        nparr_ft_2 = np.reshape(nparr_ft_2,(dataset_size,n_rows,n_cols,n_depths)) 
                        nparr_demo_2 = np.frombuffer(arr_demo_2.get_obj())
                        nparr_demo_2 = np.reshape(nparr_demo_2,(dataset_size,12,1)) 
                        nparr_la_2 = np.frombuffer(arr_la_2.get_obj())
                        nparr_la_2 = np.reshape(nparr_la_2,(dataset_size,n_labels))
                        history = model.fit([nparr_ft_2,nparr_demo_2],nparr_la_2,
                                            validation_split=validation_split, batch_size=batch_size,
                                            epochs=epochs,callbacks=callback_list,verbose=0)
                        print("fit_train_ml end -- {}".format(time.time()-start_time))
                        sys.stdout.flush()
                        acc_list.extend(history.history['acc'])
                        val_acc_list.extend(history.history['val_acc'])
                        loss_list.extend(history.history['loss'])
                        val_loss_list.extend(history.history['val_loss'])
                        turn_flag = 1
        time.sleep(1)
        n_calculation = n_calculation-1
    #save the accuracy and loss graph
    save_acc_loss_graph(acc_list,val_acc_list,loss_list,val_loss_list,input_dir+'/model_save/')
    eval_trainset(model, input_dir+'/model_save/',input_dir+'/test/')

def save_acc_loss_graph(acc_list,val_acc_list,loss_list,val_loss_list,file_path=None):
    #fig = plt.figure(1)
    fig = plt.figure()
    
    plt.subplot(121)
    plt.plot(acc_list)
    plt.plot(val_acc_list)
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='lower right')

    plt.subplot(122)
    plt.plot(loss_list)
    plt.plot(val_loss_list)
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train','test'],loc='upper right')

    fig.set_size_inches(12., 6.0)
    if file_path is not None:
        fig.savefig(file_path+'acc_and_loss graph.png',dpi=100)
    else :
        plt.show()

def eval_trainset(model, model_path, testset_path):
    re_hdf = re.compile(".*\.hdf5$")
    acc_df = pd.DataFrame(columns=['filename','acc'])
    
    best_acc_file = np.sort([x for x in os.listdir(model_path) if 'hdf5' in x])[-1]
    model.load_weights(model_path+'/'+best_acc_file)

    features, demos, labels = batch_maker(600, testset_path,shuffling=False)
    labels_pred = model.predict([features,demos])
    score = model.evaluate([features,demos],labels, verbose=0)    
    with open(model_path + '/result_report.md','w') as f:
        f.write("Evaluate testset!")
        f.write("--------------------------------------------------------------")
        f.write("%s: %.4f%%" % (model.metrics_names[1], score[1]*100))
        f.write("roc_auc_score : {}".format(roc_auc_score(Y,Y_pred)))
        f.write("--------------------------------------------------------------")
        for label_num in range(0,3):
            features, demos, labels = label_generator(testset_path,label_num=label_num,batch_size=1000)
            score = model.evaluate([features,demos],labels, verbose=0)
            f.write("label  {} testset")
            f.write("--------------------------------------------------------------")
            f.write("%s: %.4f%%" % (model.metrics_names[1], score[1]*100))
            f.write("--------------------------------------------------------------")

        for line in f.readlines():
            print(line)

def do_work_ml(args):
    ##setting in this function 
    dataset_size = 30000
    batch_size = 512
    epochs = 50
    n_rows = int(args.r)
    n_cols = int(args.c)
    n_depths = int(args.d)
    n_labels = int(args.l)
    n_calculation = 15
    data_shape = (dataset_size, n_rows, n_cols, n_depths, n_labels)
    input_dir = 'data/dataset_{}/'.format(args.t)

    #initializing model
    model = prediction_second_model(emr_rows, emr_cols, emr_depths, num_classes)

    arr_ft_1 = Array(ctypes.c_double,dataset_size*n_rows*n_cols*n_depths)
    arr_demo_1 = Array(ctypes.c_double,dataset_size*12)
    arr_la_1 = Array(ctypes.c_double,dataset_size*n_labels)
    arr_ft_2 = Array(ctypes.c_double,dataset_size*n_rows*n_cols*n_depths)
    arr_demo_2 = Array(ctypes.c_double,dataset_size*12)
    arr_la_2 = Array(ctypes.c_double,dataset_size*n_labels)

    p_t = Process(target=train_generator_ml,
        kwargs={
        'dataset_size':dataset_size,
        'arr_ft_1' : arr_ft_1,
        'arr_demo_1' : arr_demo_1,
        'arr_la_1' : arr_la_1,
        'arr_ft_2' : arr_ft_2,
        'arr_demo_2' : arr_demo_2,
        'arr_la_2' : arr_la_2,
        'n_calculation' :n_calculation,
        'input_dir' : input_dir
        })

    p_f = Process(target=fit_train_ml,
             kwargs={
             'model': model, 
             'data_shape': data_shape,
             'arr_ft_1' : arr_ft_1,
             'arr_demo_1' : arr_demo_1,
             'arr_la_1' : arr_la_1,
             'arr_ft_2' : arr_ft_2,
             'arr_demo_2' : arr_demo_2,
             'arr_la_2' : arr_la_2,
             'n_calculation':n_calculation,
             'epochs' : epochs,
             'input_dir' : input_dir
             })

    p_t.start()
    time.sleep(10) 
    #process_train_generator 와 process_fit_train의 연산 order을 맞춰주기 위해서， 임의로 넣은 sleep
    p_f.start()
    p_t.join()
    p_f.join()
    

def _set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('t', help="dataset number")
    parser.add_argument('r', help="emr row")
    parser.add_argument('c', help="emr col")
    parser.add_argument('d', help="emr depth")
    parser.add_argument('l', help="the number of label")
    args = parser.parse_args()
    return args


if __name__=='__main__':
    args = _set_parser()
    do_work_ml(args)
