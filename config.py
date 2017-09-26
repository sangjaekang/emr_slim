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

