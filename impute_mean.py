# -*- coding: utf-8 -*-
import sys, os, re
import pandas as pd
import numpy as np
from construct_prescribe import get_prescribe_df

os_path = os.path.abspath('./') ; find_path = re.compile('emr_slim')
BASE_PATH = os_path[:find_path.search(os_path).span()[1]]
sys.path.append(BASE_PATH)

from config import *
'''
응급코드와　비응급코드는　결과값을　공유
|측정내용|응급코드|비응급코드|
|----|----|----|
|총단백|L8031|L3011|
|알부민|L8032|L3012|
|당정량|L8036|L3013|
|요소질소|L8037|L3018|
|크레이타닌|L8038|L3020|
|소디움|L8041|L3041|
|포타슘|L8042|L3042|
|염소 |L8043|L3043|
|혈액탄산|L8044|L3044|
|총칼슘|L8046|L3022|
|인   |L8047|L3023|
|요산|L8048|L3021|
|총콜레스테롤|L8049|L3013|
|중성지방|L8050|L3029|
|LDH|L8053|L3057|
'''

EMGCY_AND_NOT_DICT = {
    'L8031':'L3011',   'L8032':'L3012',
    'L8036':'L3013',   'L8037':'L3018',
    'L8038':'L3020',   'L8041':'L3041',
    'L8042':'L3042',   'L8043':'L3043', 
    'L8044':'L3044',   'L8046':'L3022',
    'L8047':'L3023',   'L8048':'L3021',
    'L8049':'L3013',   'L8050':'L3029', 'L8053':'L3057'
}

# 반복적으로　이용되는　값을　전역으로　해둠
lab_store = pd.HDFStore(labtest_output_path,mode='r')
LAB_INDEX = lab_store.select('metadata/usecol').col.values
lab_store.close()

def get_labtest_avg_map():
    global labtest_output_path

    if not os.path.isfile(labtest_output_path):
        raise ValueError("There is no labtest_OUTPUT file!")

    labtest_mapping_df = pd.HDFStore(labtest_output_path,mode='r').select('metadata/mapping_table').set_index('labtest')
    avg_map = (labtest_mapping_df.AVG -labtest_mapping_df.MIN)/(labtest_mapping_df.MAX-labtest_mapping_df.MIN)
    return avg_map

# 반복적으로　이용되는　값을　전역으로　해둠
LAB_AVG_MAP= get_labtest_avg_map()

def _mean_with_nan(x,y):
    len_x = len(x.values)
    result_series = pd.Series(index=x.index)
    
    for i in range(len_x):
        if np.isnan(x.values[i]):
            if np.isnan(y.values[i]):
                result_series.iloc[i] = np.nan
            else:
                result_series.iloc[i] = y.values[i]
        else:
            if np.isnan(y.values[i]):
                result_series.iloc[i] = x.values[i]
            else:
                result_series.iloc[i] = np.mean((x.values[i],y.values[i]))
    return result_series

def _nan_or_not(x):
    return 0 if np.isnan(x) else 1

def _suffle_time(x): 
    return np.int(np.floor(np.random.normal(scale=x)))

def get_np_bool_emr(np_array):
    nan_or_not = np.vectorize(_nan_or_not,otypes=[np.float])
    bool_array = nan_or_not(np.array(np_array,copy=True))
    return bool_array

def get_np_imputation_emr(np_array):
    global LAB_INDEX, LAB_AVG_MAP
    result_array = np.array(np_array,copy=True)

    for i in range(result_array.shape[0]):
        inds = np.argwhere(~np.isnan(result_array[i,:]))
        if inds.size == 0: 
            result_array[i,:] = LAB_AVG_MAP[LAB_INDEX[i]]
        elif inds.size == 1:
            result_array[i,:] = result_array[i,inds[0,0]]
        else:
            prev_ind = None
            for ind in inds[:,0]:
                if prev_ind is not None:
                    prev_value = result_array[i,prev_ind]
                    curr_value = result_array[i,ind]
                    for input_index in range(prev_ind,ind+1):
                        result_array[i,input_index] = \
                        (curr_value-prev_value)/(ind-prev_ind)*(input_index-prev_ind)+prev_value
                prev_ind = ind
            result_array[i,:inds[:,0][0]] = result_array[i,inds[:,0][0]]
            result_array[i,inds[:,0][-1]:] = result_array[i,inds[:,0][-1]]
    return result_array

def get_np_array_pres(input_path,width=12):
    no, order = input_path.split('/')[-1].replace('.npy',"").split('_')[0], input_path.split('/')[-1].replace('.npy',"").split('_')[1]
    prescribe_df = get_prescribe_df(no)
    window = prescribe_df.iloc[:,int(order):int(order)+width]
    del prescribe_df
    return window.as_matrix()

def get_np_array_emr(input_path,shuffling=True):
    global LAB_INDEX, EMGCY_AND_NOT_DICT
    
    # 50%, not shuffling
    if np.random.randint(0,2):
        shuffling=False

    np_array = np.load(input_path).astype(float)
    # 응급과　비응급　코드의　값을　공유
    bool_array = get_np_bool_emr(np_array[0]) # get boolean mask
    
    result_list = []
    for index in range(np_array.shape[0]):
        temp_array = pd.DataFrame(index=LAB_INDEX,data=np_array[index])   
        for emg,not_emg in EMGCY_AND_NOT_DICT.items():
            avg_test = _mean_with_nan(temp_array.loc[emg],temp_array.loc[not_emg])
            temp_array.loc[emg] = avg_test
            temp_array.loc[not_emg] = avg_test
        result_list.append(temp_array.as_matrix())
    np_array = np.stack(result_list,axis=0)

    # shuffling time for data augumentation
    result_array = np.full(np_array.shape,np.nan)
    if shuffling:
        r_time = np_array.shape[2]
        for x,y in np.argwhere(~np.isnan(np_array[0])):
            m_y = y+_suffle_time(1)
            while (m_y<0)|(m_y>=r_time): m_y = y+_suffle_time(1)
            for index in range(np_array.shape[0]):
                result_array[:,x,m_y] = np_array[:,x,y] + np.random.normal(0,0.01) # add gaussian noise
    else:
        result_array = np.array(np_array,copy=True)

    result_list = []
    result_list.append(bool_array)
    for i in range(result_array.shape[0]):
        imput_array = get_np_imputation_emr(result_array[i]) # get imputation mask
        result_list.append(imput_array)

    result = np.stack(result_list,axis=-1)
    del bool_array, imput_array, np_array, result_list
    return result