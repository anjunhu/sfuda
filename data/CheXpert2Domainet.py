import h5py
import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
import pickle
import time
from tqdm import tqdm

DATA_DIR = '/home/shared_space/data/CheXpert-v1.0-small/'
IGNORE_EMPTY_ENTRIES = False
VERBOSE = False
DOMAIN = 'chexpert'
ATTRIBUTES = ['Atelectasis' ,'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

for TARGET_ATTRIBUTE in ATTRIBUTES:
    for SPLIT in ['']: #['train', 'valid']:
        # read metadata
        DATA_DIR = '/home/shared_space/data/CheXpert-v1.0-small/'
        train_df = pd.read_csv(DATA_DIR + f'train.csv')
        valid_df = pd.read_csv(DATA_DIR + f'valid.csv') # Path,Sex,Age,Frontal/Lateral,AP/PA
        dmgph_df = pd.read_csv(f'CHEXPERT_DEMO.csv') # PATIENT,GENDER,AGE_AT_CXR,PRIMARY_RACE,ETHNICITY
        #print(pd.unique(dmgph_df['PRIMARY_RACE']))
        merged_df = pd.concat([train_df, valid_df])
        merged_df.reset_index(inplace=True)
        #merged_df = merged_df.head(100)

        remover = []
        GENDERs = []
        AGE_AT_CXRs = []
        PRIMARY_RACEs = []
        ETHNICITYs = []
        for i in tqdm(range(len(merged_df))):
            pid = merged_df.iloc[i]['Path'].split('/')[2]
            row_in_dmgph_df = dmgph_df.loc[dmgph_df['PATIENT'] == pid]
            if len(row_in_dmgph_df) != 1:
                remover.append(i)
                continue
            #assert(row_in_dmgph_df['AGE_AT_CXR'].item()==merged_df.iloc[i]['Age'])
            if row_in_dmgph_df['GENDER'].item() != merged_df.iloc[i]['Sex']:
                print(row_in_dmgph_df['GENDER'], merged_df.iloc[i]['Sex'])
                remover.append(i)
                continue
            GENDERs.append(row_in_dmgph_df['GENDER'])
            AGE_AT_CXRs.append(row_in_dmgph_df['AGE_AT_CXR'])
            PRIMARY_RACEs.append(row_in_dmgph_df['PRIMARY_RACE'])
            ETHNICITYs.append(row_in_dmgph_df['ETHNICITY'])

        merged_df.drop(remover, inplace=True)
        assert(len(merged_df) == len(GENDERs))
        assert(len(merged_df) == len(AGE_AT_CXRs))
        assert(len(merged_df) == len(PRIMARY_RACEs))
        assert(len(merged_df) == len(ETHNICITYs))
        merged_df['AGE_AT_CXR'] = AGE_AT_CXRs
        merged_df['PRIMARY_RACE'] = PRIMARY_RACEs
        merged_df['ETHNICITY'] = ETHNICITYs
        merged_df['GENDER'] = GENDERs

        # unify the value of sensitive attributes
        sex = merged_df['Sex'].values
        sex[sex == 'Male'] = 0
        sex[sex == 'Female'] = 1
        merged_df['Sex'] = sex
        ta = merged_df[TARGET_ATTRIBUTE].values
        print(TARGET_ATTRIBUTE, 'ONES', 'ZEROS', 'NEGONES', 'NANS')
        print(len(ta[ta==1]), len(ta[ta==0]), len(ta[ta==-1]), len(ta[ta!=ta]))
        print(len(ta[ta==1])/len(ta), len(ta[ta==0])/len(ta), len(ta[ta==-1])/len(ta), len(ta[ta!=ta])/len(ta))

        # Ignore Uncertain and Emtpy entires
        if IGNORE_EMPTY_ENTRIES:
            merged_df = merged_df[~merged_df[TARGET_ATTRIBUTE].isnull()]
            assert(len(ta[ta!=ta])==0)
        merged_df = merged_df[merged_df[TARGET_ATTRIBUTE]!=-1.0]
        ta = merged_df[TARGET_ATTRIBUTE].values
        assert(len(ta[ta==-1])==0)
        ta[ta != 1.0] = 0.0
        merged_df[TARGET_ATTRIBUTE] = ta

        # Null to negative
        ta = merged_df[TARGET_ATTRIBUTE].values.copy()
        ta[ta != ta] = '0.0'
        ta = ta.astype('int')
        merged_df[TARGET_ATTRIBUTE] = ta
        merged_df['binaryLabel'] = ta

        # split subjects to different age groups
        merged_df['Age_multi'] = merged_df['AGE_AT_CXR'].values.astype('int')
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(-1,19), 0, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(20,39), 1, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(40,59), 2, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(60,79), 3, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi']>=80, 4, merged_df['Age_multi'])

        merged_df['Age_binary'] = merged_df['AGE_AT_CXR'].values.astype('int')
        merged_df['Age_binary'] = np.where(merged_df['Age_binary'].between(-1, 60), 0, merged_df['Age_binary'])
        merged_df['Age_binary'] = np.where(merged_df['Age_binary']>= 60, 1, merged_df['Age_binary'])

        merged_df['Sex_binary'] = merged_df['Sex']
        merged_df['Sex_binary'] = np.where(merged_df['Sex_binary']=='M', 0, merged_df['Sex_binary'])
        merged_df['Sex_binary'] = np.where(merged_df['Sex_binary']=='F', 1, merged_df['Sex_binary'])

        '''
        ['Other' 'White, non-Hispanic' 'Black or African American' 'White'
         'Native Hawaiian or Other Pacific Islander' 'Asian' 'Asian, non-Hispanic'
         'Unknown' 'Native American, non-Hispanic' 'Race and Ethnicity Unknown'
         'White, Hispanic' nan 'Other, Hispanic' 'Black, non-Hispanic'
         'American Indian or Alaska Native' 'Patient Refused'
         'Other, non-Hispanic' 'Pacific Islander, Hispanic' 'Black, Hispanic'
         'Pacific Islander, non-Hispanic' 'White or Caucasian' 'Asian, Hispanic'
         'Native American, Hispanic' 'Asian - Historical Conv']
        '''
        merged_df['PRIMARY_RACE'] = merged_df['PRIMARY_RACE'].astype(str)
        merged_df['Race'] = merged_df['PRIMARY_RACE'].astype(str)
        merged_df['Race'] = np.where(merged_df['PRIMARY_RACE'].str.contains("Black"), 0, merged_df['Race'])
        merged_df['Race'] = np.where(merged_df['PRIMARY_RACE'].str.contains("White"), 1, merged_df['Race'])
        merged_df['Race'] = np.where(merged_df['PRIMARY_RACE'].str.contains("Asian"), 2, merged_df['Race'])
        merged_df = merged_df[merged_df.Race.isin([0, 1, 2])]

        # A peek at post-binarization training set P(A), P(Y), P(Y, A)
        ta = TARGET_ATTRIBUTE.replace(' ', '')
        print(ta, 'AGE_AT_CXR', 'Y0', 'Y1', 'Total')
        for sa in range(5):
            df_sa = merged_df[merged_df['Age_multi'] == sa]
            df_y0 = merged_df[(merged_df['binaryLabel']==0) & (merged_df['Age_multi'] == sa)]
            df_y1 = merged_df[(merged_df['binaryLabel']==1) & (merged_df['Age_multi'] == sa)]
            print(f'A{sa}', len(df_y0), len(df_y1), len(df_sa))
        print(ta, 'Sex', 'Y0', 'Y1', 'Total')
        for sa in range(2):
            df_sa = merged_df[merged_df['Sex_binary'] == sa]
            df_y0 = merged_df[(merged_df['binaryLabel']==0) & (merged_df['Sex_binary'] == sa)]
            df_y1 = merged_df[(merged_df['binaryLabel']==1) & (merged_df['Sex_binary'] == sa)]
            print(f'A{sa}', len(df_y0), len(df_y1), len(df_sa))
        print(ta, 'Race', 'Y0', 'Y1', 'Total')
        for sa in range(3):
            df_sa = merged_df[merged_df['Race'] == sa]
            df_y0 = merged_df[(merged_df['binaryLabel']==0) & (merged_df['Race'] == sa)]
            df_y1 = merged_df[(merged_df['binaryLabel']==1) & (merged_df['Race'] == sa)]
            print(f'A{sa}', len(df_y0), len(df_y1), len(df_sa))

        lines = []
        start = time.time()
        for i in tqdm(range(len(merged_df))):
            os.makedirs(os.path.join('domainnet_style_datasets', ta, DOMAIN, str(int(merged_df.iloc[i]['binaryLabel']))), exist_ok=True) 
            img = cv2.imread(os.path.join(DATA_DIR, "..", merged_df.iloc[i]['Path']))
            # resize to the input size in advance to save time during training
            img = cv2.resize(img, (256, 256))
            filename = merged_df.iloc[i]['Path'].split("/")
            filename = os.path.join(DOMAIN, str(int(merged_df.iloc[i]['binaryLabel'])), '_'.join(filename[-3:]))
            lines.append(filename+' '+str(int(merged_df.iloc[i]['binaryLabel']))+' '+str(int(merged_df.iloc[i]['Age_multi']))+\
                                  ' '+str(int(merged_df.iloc[i]['Sex_binary']))+' '+str(int(merged_df.iloc[i]['Race'])))
            #print(lines[-1])
            cv2.imwrite(os.path.join(f'domainnet_style_datasets', ta, filename), img)
        end = time.time()
        print('Time Elapsed', end-start)

        with open(f'domainnet_style_datasets/{ta}/chexpert_list.txt', 'w') as f:
            f.write('\n'.join(lines))
