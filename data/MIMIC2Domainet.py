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

DATA_DIR = '/scratch/local/ssd/anjun/datasets/physionet.org/files/mimic-cxr-jpg/2.0.0/'
IGNORE_EMPTY_ENTRIES = False
VERBOSE = True
DOMAIN = 'mimic'
ATTRIBUTES = ['Atelectasis' ,'Cardiomegaly', 'Consolidation', 'Edema', 'Pleural Effusion']

for TARGET_ATTRIBUTE in ATTRIBUTES:
        # read dataframes
        labels_df = pd.read_csv(DATA_DIR + 'mimic-cxr-2.0.0-chexpert.csv')
        splits_df = pd.read_csv(DATA_DIR + 'mimic-cxr-2.0.0-split.csv')
        metadata_df = pd.read_csv(DATA_DIR + 'mimic-cxr-2.0.0-metadata.csv')
        admissions_df = pd.read_csv(DATA_DIR + 'admissions.csv')
        patients_df = pd.read_csv(DATA_DIR + 'patients.csv')

        # remove patients with inconsistent race information (github.com/robintibor)
        ethnicity_df = admissions_df.drop_duplicates()
        ethnicity_df = ethnicity_df[['subject_id', 'race']]
        v = ethnicity_df.subject_id.value_counts()
        subject_id_more_than_once = v.index[v.gt(1)]
        ambiguous_ethnicity_df = ethnicity_df[ethnicity_df.subject_id.isin(subject_id_more_than_once)]
        inconsistent_race = ambiguous_ethnicity_df.subject_id.unique()
        #grouped = ambiguous_ethnicity_df.groupby('subject_id')
        #grouped.aggregate(lambda x: "_".join(sorted(x))).race.value_counts()

        merged_df = pd.merge(metadata_df,labels_df,on=['subject_id', 'study_id'])
        merged_df = pd.merge(ethnicity_df,merged_df,on=['subject_id'])
        merged_df = pd.merge(patients_df,merged_df,on=['subject_id'])
        merged_df = pd.merge(splits_df,merged_df,on=['subject_id', 'study_id', 'dicom_id'])
        merged_df = merged_df[~merged_df.subject_id.isin(inconsistent_race)]
        merged_df = merged_df[merged_df.race.isin(['ASIAN','BLACK/AFRICAN AMERICAN','WHITE'])]
        merged_df = merged_df[merged_df.ViewPosition.isin(['AP','PA'])]
        merged_df.reset_index(inplace=True)
        if VERBOSE:
            print("Total images after inclusion/exclusion criteria: " + str(len(merged_df)))
            print("Total patients after inclusion/exclusion criteria: " + str(merged_df.subject_id.nunique()))
            #print(pd.unique(merged_df['race']))
            print(merged_df.index)

        start = time.time()

        images = []
        paths = []
        binaryLabels = []
        path = '/home/shared_space/data/physionet.org/files/mimic-cxr-jpg/2.0.0/files/'
        remover = []
        for i in tqdm(range(len(merged_df))):
            p2jpg = os.path.join(DATA_DIR, 'files', 'p'+str(merged_df.iloc[i]['subject_id'])[:2], 'p'+str(merged_df.iloc[i]['subject_id']),
                                          's'+str(merged_df.iloc[i]['study_id']), str(merged_df.iloc[i]['dicom_id'])+'.jpg')
            if not os.path.isfile(p2jpg):
                remover.append(i)
                #print(f'Dropping entry {i} due to missing file!')
                continue
            img = cv2.imread(os.path.join(DATA_DIR, 'files', 'p'+str(merged_df.iloc[i]['subject_id'])[:2], 'p'+str(merged_df.iloc[i]['subject_id']),
                                          's'+str(merged_df.iloc[i]['study_id']), str(merged_df.iloc[i]['dicom_id'])+'.jpg'))
            # resize to the input size in advance to save time during training
            img = cv2.resize(img, (256, 256))
            images.append(img)
            paths.append(p2jpg)
            binaryLabels.append(float(merged_df.iloc[i][TARGET_ATTRIBUTE]))

        merged_df.drop(remover, inplace=True)
        assert(len(merged_df) == len(images))
        merged_df['image'] = images
        merged_df['path'] = paths
        merged_df[TARGET_ATTRIBUTE] = binaryLabels
        #print(merged_df)

        # Get some pre-binarization P(Y) stats
        ta = merged_df[TARGET_ATTRIBUTE].values
        if VERBOSE:
            print('\n\n\n', TARGET_ATTRIBUTE, 'ONES', 'ZEROS', 'NEGONES', 'NANS')
            print(len(ta[ta==1]), len(ta[ta==0]), len(ta[ta==-1]), len(ta[ta!=ta]))
            print(len(ta[ta==1])/len(ta), len(ta[ta==0])/len(ta), len(ta[ta==-1])/len(ta), len(ta[ta!=ta])/len(ta))

        # Binarize Target Attribute
        # Option 1: Null-to-Negative
        # Ignore Uncertain -1 and Emtpy NaN entires
        if IGNORE_EMPTY_ENTRIES:
            merged_df = merged_df[~merged_df[TARGET_ATTRIBUTE].isnull()]
            assert(len(ta[ta!=ta])==0)
        merged_df = merged_df[merged_df[TARGET_ATTRIBUTE]!=-1.0]
        ta = merged_df[TARGET_ATTRIBUTE].values
        assert(len(ta[ta==-1])==0)
        ta[ta != 1.0] = 0.0
        merged_df[TARGET_ATTRIBUTE] = ta

        # Option 2: Drop Nulls
        ta = merged_df[TARGET_ATTRIBUTE].values.copy()
        ta[ta != ta] = '0.0'
        ta = ta.astype('int')
        merged_df[TARGET_ATTRIBUTE] = ta
        merged_df['binaryLabel'] = ta

        # MEDFAIR-style Sensitive Attributes
        merged_df['Age_multi'] = merged_df['anchor_age'].values.astype('int')
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(-1,19), 0, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(20,39), 1, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(40,59), 2, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi'].between(60,79), 3, merged_df['Age_multi'])
        merged_df['Age_multi'] = np.where(merged_df['Age_multi']>=80, 4, merged_df['Age_multi'])
        merged_df['Age_binary'] = merged_df['anchor_age'].values.astype('int')
        merged_df['Age_binary'] = np.where(merged_df['Age_binary'].between(-1, 60), 0, merged_df['Age_binary'])
        merged_df['Age_binary'] = np.where(merged_df['Age_binary']>= 60, 1, merged_df['Age_binary'])
        merged_df['Sex_binary'] = merged_df['gender']
        merged_df['Sex_binary'] = np.where(merged_df['Sex_binary']=='M', 0, merged_df['Sex_binary'])
        merged_df['Sex_binary'] = np.where(merged_df['Sex_binary']=='F', 1, merged_df['Sex_binary'])
        merged_df['Race'] = merged_df['race']
        merged_df['Race'] = np.where(merged_df['Race']=="BLACK/AFRICAN AMERICAN", 0, merged_df['Race'])
        merged_df['Race'] = np.where(merged_df['Race']=="WHITE", 1, merged_df['Race'])
        merged_df['Race'] = np.where(merged_df['Race']=="ASIAN", 2, merged_df['Race'])

        end = time.time()
        if VERBOSE: print('Time Elapsed: ', end-start)
        ta = TARGET_ATTRIBUTE.replace(' ', '')
        # Option 1: DIY splits to have controllabe val size and consistent P(Y), P(A), P(Y, A) in training and validation sets
        merged_df_train, merged_df_valid = train_test_split(merged_df, test_size=0.2)
        # Option 2: Use official splits
        #sp = merged_df['split'].values
        #merged_df_train = merged_df[sp=='train']
        #merged_df_val = merged_df[sp=='validate']
        print(ta, 'train size', len(merged_df_train), 'val size', len(merged_df_valid))

        # A peek at post-binarization training set P(A), P(Y), P(Y, A)
        print(ta, 'Age', 'Y0', 'Y1', 'Total')
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
        path = '/home/shared_space/data/'
        for i in tqdm(range(len(merged_df))):
            os.makedirs(os.path.join('domainnet_style_datasets', ta, DOMAIN, str(int(merged_df.iloc[i]['binaryLabel']))), exist_ok=True) 
            img = merged_df.iloc[i]['image']
            filename = merged_df.iloc[i]['path']
            filename = '_'.join(filename.split("/")[-4:])
            filename = os.path.join(DOMAIN, str(int(merged_df.iloc[i]['binaryLabel'])), filename)
            lines.append(filename+' '+str(int(merged_df.iloc[i]['binaryLabel']))+' '+str(int(merged_df.iloc[i]['Age_multi']))+\
                                  ' '+str(int(merged_df.iloc[i]['Sex_binary']))+' '+str(int(merged_df.iloc[i]['Race'])))
            #if VERBOSE: print(lines[-1])
            cv2.imwrite(os.path.join('domainnet_style_datasets', ta, filename), img)
        end = time.time()
        if VERBOSE: print('Time Elapsed', end-start)

        with open(f'domainnet_style_datasets/{ta}/{DOMAIN}_list.txt', 'w') as f:
            f.write('\n'.join(lines))
