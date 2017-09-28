# -*- coding: utf-8 -*-
import re
import os
os_path = os.path.abspath('./')
find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]

## DATA Setting
DELIM = '\x0b'
LAB_COL_NAME = ['no','lab_code','date','result']
USE_LAB_COL_NAME = ['no','date','result']

## Debug
DEBUG_PRINT = True

## time period among data
START_TIME = 20100101
END_TIME  = 20161231

## data directory
DATA_DIR  = BASE_PATH + '/data/'
PREP_OUTPUT_DIR  = DATA_DIR + 'prep/'
INPUT_DIR = DATA_DIR +'input/'

DEMOGRAPHIC_OUTPUT_PATH = 'demo_df.h5'
LABEL_PATIENT_PATH = 'label_patient_df.h5'
LABTEST_OUTPUT_PATH = 'labtest_df.h5'
PRESCRIBE_OUTPUT_PATH = 'prescribe_df.h5'

prescribe_output_path = PREP_OUTPUT_DIR + PRESCRIBE_OUTPUT_PATH
labtest_output_path = PREP_OUTPUT_DIR + LABTEST_OUTPUT_PATH

def check_directory(directory_path):
    # syntax checking for directory
    if not (directory_path[-1] is '/'):
        directory_path  = directory_path + '/'
    # not exists in directory
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
    
    return directory_path        


def yield_time_column(time,end_t):
    # time-series column을　만드는데　있어서　순차적으로　생성하는　함수
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


def check_time_format(x):
    # timeformat(ex: 20170801)의　형태로　되어있는지　확인하는　함수
    re_time = re.compile('^\d{8}$')
    if re_time.match(str(x)):
        return int(str(x)[2:6])
    else:
        raise ValueError("wrong type time format : {}".format(x))


def get_time_interval(t_1,t_2):
    #t_1 t_2 두 시각의 시간 차 계산
    def _get_time(t):
        year = t//100; month = t%100
        return (year*12+month)
    if t_1<t_2:
        return _get_time(t_2)-_get_time(t_1)
    else :
        return _get_time(t_1)-_get_time(t_2)


def get_add_interval(t,interval):
    #t에다가　interval을　더한　시간
    year = t//100; month = t%100
    times = (year*12+month+interval)
    return (times//12*100+ times%12)


def convert_month(x):
    '''
    datetype을　month 단위로　바꾸어주는　함수
        ex) 20110132 -> 1101
    '''
    re_date = re.compile('^\d{8}$') 

    str_x = str(x)
    if re_date.match(str_x):
        return int(str_x[2:6])
    else : 
        raise ValueError("wrong number in date : {}".format(str_x))

def convert_times_per_month(x):
    return float(x) // 30
