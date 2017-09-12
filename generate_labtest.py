# -*- coding: utf-8 -*-
import sys, os, re
import numpy as np
import pandas as pd
import time

os_path = os.path.abspath('./') ; find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from construct_labtest import get_labtest_df

DEBUG_PRINT=True

## time period among data
START_TIME = 20100101
END_TIME  = 20161231

## data directory
DATA_DIR  = BASE_PATH + '/data/'
PREP_OUTPUT_DIR  = DATA_DIR + 'prep/'
INPUT_DIR = DATA_DIR +'input/'
LABEL_PATIENT_PATH = 'label_patient_df.h5'


def save_patient_input(no_range,label_name,save_dir=None,time_length=6,gap_length=1,target_length=3,offset_min_counts=50,offset_max_counts=100):
    global save_dir, DEBUG_PRINT, PREP_OUTPUT_DIR, LABEL_PATIENT_PATH
    if save_dir is None:
        save_dir = INPUT_DIR

    for label_value in range(0,4):
        o_path = save_dir +'{}/'.format(label_value)
        if not os.path.isdir(o_path):
            os.makedirs(o_path)

    # syntax checking existence for directory
    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH
    label_df = pd.HDFStore(output_path,mode='r').select('label/{}'.format(label_name))

    colist = get_timeseries_column()

    for idx, no in enumerate(no_range):
        if DEBUG_PRINT and idx %1000 == 0:  print("process({}){} th start".format(os.getpid(),idx))
        emr_df = get_labtest_df(no)
        label_series = get_patient_timeseries_label(no,label_df)

        for i in range(0,len(colist) - (time_length+gap_length+target_length)):
            window = emr_df.loc[:,colist[i]:colist[time_length-1+i]]
            if window.count().sum() >= offset_min_counts and window.count().sum() <= offset_max_counts:
                target_stime = colist[time_length-1+gap_length+i]
                label_value = check_label(label_series.loc[target_stime:target_stime+target_length-1])
                if np.isnan(label_value): continue
                file_path = save_dir +'{}/{}_{}.npy'.format(label_value,no,i)
                np.save(file_path,window.as_matrix())


def get_patient_timeseries_label(no,label_df):
    global  PREP_OUTPUT_DIR,  LABEL_PATIENT_PATH
    result_series = pd.Series(index=get_timeseries_column())    
    for _,label_df_date,label_df_label in label_df[label_df.no==no].values:
        result_series[label_df_date]=label_df_label
    
    return result_series


def check_directory(directory_path):
    # syntax checking for directory
    if not (directory_path[-1] is '/'):
        directory_path  = directory_path + '/'
    # not exists in directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    return directory_path        


def yield_time_column(time,end_t):
    # time-series column을　만드는데　있어서　순차적으로　생성하는　함수X
    # 1210 1211 1212 1301 1302 이런식으로　생성
    while time <= end_t:
        yield time
        year = time // 100 ; month = time % 100
        if month >= 12 :
            year = year + 1
            month = 1
        else : 
            month = month + 1
        time = (year*100 + month) 
        

def get_timeseries_column():
    global START_TIME, END_TIME
    start_time = check_time_format(START_TIME)
    end_time  = check_time_format(END_TIME)

    col_list = []
    for time in yield_time_column(start_time,end_time):
        col_list.append(time)

    return col_list


def get_time_interval(t_1,t_2):
    #t_1 t_2 두 시각의 시간 차 계산
    def _get_time(t):
        year = t//100; month = t%100
        return (year*12+month)
    if t_1<t_2:
        return _get_time(t_2)-_get_time(t_1)
    else :
        return _get_time(t_1)-_get_time(t_2)


def write_metadata_README(path, label_name,time_length,gap_length,target_length,offset_min_counts,offset_max_counts):
    metadata_README = '''
### parameter Setting
| parameter             | value |
| ---------------------    | ----- |
| label_name            | {}    |
| time_length           | {}    |
| gap_length            | {}    |
| target_length         | {}    |
| offset_min_counts | {}    |
| offset_max_counts | {}    |
| Created date          | {}    |
'''.format(label_name,time_length,gap_length,target_length,offset_min_counts,offset_max_counts,time.ctime())
    
    file_path = path + 'README.md'    
    with open(file_path,'w') as f:
        f.write(metadata_README)


def check_label(x):
    NORMAL_FLAG = False
    for _,y in x.items():
        if y>0:  return int(y)
            # hyper : y==2
            # hypo  : y==1
        if y is 0 : NORMAL_FLAG = True
            # normal : y==0
    if NORMAL_FLAG:
        return 0
    else:
        return np.nan


def _set_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='save path')
    parser.add_argument('label', help='label_name')
    parser.add_argument('time_length',help='time_length')
    parser.add_argument('gap_length', help='gap_length')
    parser.add_argument('target_length',help='target_length')
    parser.add_argument('offset_min_counts',help='offset_min_counts')
    parser.add_argument('offset_max_counts',help='offset_max_counts')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    global PREP_OUTPUT_DIR, LABEL_PATIENT_PATH
    #argument
    args = _set_parser()
    label_name = args.label 
    time_length = int(args.time_length) 
    gap_length = int(args.gap_length)
    target_length = int(args.target_length)
    offset_min_counts = int(args.offset_min_counts)
    offset_max_counts = int(args.offset_max_counts) 
    
    o_path = check_directory(args.path)

    train_path = o_path + 'train/'; train_path = check_directory(train_path)
    test_path = o_path + 'test/'; test_path = check_directory(test_path)
    validation_path = o_path + 'validation/'; validation_path = check_directory(validation_path)

    PREP_OUTPUT_DIR = check_directory(PREP_OUTPUT_DIR)
    output_path = PREP_OUTPUT_DIR + LABEL_PATIENT_PATH

    write_metadata_README(o_path, label_name,time_length,gap_length,target_length,offset_min_counts,offset_max_counts)

    sample_store = pd.HDFStore(output_path,mode='r')
    train_set = sample_store.select('label/{}/train'.format(label_name)).no.unique()
    validation_set = sample_store.select('label/{}/validation'.format(label_name)).no.unique()
    test_set = sample_store.select('label/{}/test'.format(label_name)).no.unique()
    sample_store.close()

    ## train set generating
    print("Creating pool with 8 workers")
    start_time = time.time()
    pool = multiprocessing.Pool()
    print("Invoking apply train_set")
    for divider in np.array_split(train_set,8):
        pool.apply_async(save_patient_input,args=(divider,label_name,train_path,time_length,
                     gap_length,target_length,offset_min_counts,offset_max_counts,))
    pool.close()
    pool.join()    
    print("trainset Finished--consumed time : {}".format(time.time()-start_time))

    ## test set generating
    print("Creating pool with 8 workers")
    start_time = time.time()
    pool = multiprocessing.Pool()
    print("Invoking apply test_set")
    for divider in np.array_split(test_set,8):
        pool.apply_async(save_patient_input,args=(divider,label_name,test_path,time_length,
                     gap_length,target_length,offset_min_counts,offset_max_counts,))
    pool.close()
    pool.join()    
    print("testset Finished--consumed time : {}".format(time.time()-start_time))

    ## validation set generating
    print("Creating pool with 8 workers")
    start_time = time.time()
    pool = multiprocessing.Pool()
    print("Invoking apply test_set")
    for divider in np.array_split(validation_set,8):
        pool.apply_async(save_patient_input,args=(divider,label_name,validation_path,time_length,
                     gap_length,target_length,offset_min_counts,offset_max_counts,))
    pool.close()
    pool.join()    
    print("validation Finished--consumed time : {}".format(time.time()-start_time))
