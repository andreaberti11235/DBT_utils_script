#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 12:12:46 2021

@author: si-lab
"""

import pandas as pd
import argparse
import os
import shutil
import numpy as np
import glob

# #
# osservazione: parlando con eva 
# TODO: dovremmo tenere conto nella suddivisione train_test_split che le immagini
# appartenenti alla stessa lesione vadano TUTTE o nel training o nel test,
# invece al momento noi stiamo splittando indistintamente
# #




parse = argparse.ArgumentParser(description="At the end of this script we obtain three dataset folders: push, valid, test.\n To use original images only (not corrupted), do not pass the argument -c")

parse.add_argument('png_dir', help='Path to the input directory with original PNG images (parent folder of class folders, e.g. \n '\
                   '*dataset*, parent of *benign* and *cancer* folders')
parse.add_argument('-c','--corrupted_png_dir', help='Path to the input directory that group differently-corrupted PNG images',type=str)
parse.add_argument('-csv','--csv_file', help='Path to the CSV file of original images')
parse.add_argument('-d', '--dest_dir', help='Path to destination directory, parent folder of: test, push, valid, train datasets')

args = parse.parse_args()

png_dir = args.png_dir
corrupted_png_dir = args.corrupted_png_dir
csv_file_path = args.csv_file
dest_dir = args.dest_dir


df = pd.read_csv(csv_file_path,sep=',', index_col='file_name')

from distutils.dir_util import copy_tree
from stratified_group_data_splitting import StratifiedGroupKFold

# Copiamo la directory di test nella destination dir
# copy_tree('/media/si-lab/63bc1baf-d08c-4db5-b271-e462f3f4444d/a_e_g/datasets/breast_CBIS-DDSM/dataset_png/test', os.path.join(dest_dir, 'test'))


# Suddivisione in push e valid

X = np.array(df.index)
group = np.array(df['patient_id'])
y = np.array(df['label'])

y = np.array([0 if elem=='benign' else 1 for elem in y]) #TODO modificare con la stringa opportuna

x = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42) #reproducibility, push è 80% del training originale e valid 20% # split primo con 123, secondo split (07/04/22) con 42
for push_idxs, valid_idxs in x.split(X,y,groups=group):
    print('Prima iterazione completata, esco')
    break

push_names = X[push_idxs]
push_labels = y[push_idxs]
push_group = group[push_idxs] #sarà il nuovo array per raggruppare immagini nella successiva suddivisione da TEMP-->PUSH,VALID

valid_names = X[valid_idxs]
valid_labels = y[valid_idxs]



# VECCHIA VERSIONE CON TRAINTESTSPLIT
# stratify_col = df.iloc[:,1] #Label

# # IL VALORE stringa DELLE CELLE NEL CSV CAMBIA IN BASE AL DATASET, (de)commentare qui sotto:
    
# # Prostata: HG, LG
# #stratify_col = pd.Series([int(elem=='HG') for elem in stratify_col],index=df.index) #retain the Index column from previous dataframe

# # Breast: Malignant, Benign
# stratify_col = pd.Series([int(elem=='Malignant') for elem in stratify_col],index=df.index) #retain the Index column from previous dataframe


# from sklearn.model_selection import train_test_split
# id_push_temp,id_test = train_test_split(list(df.index),test_size=0.2,random_state=27,shuffle=True, stratify=stratify_col)
# #Per potere usare la stratificazione qui sotto devo ricomputare la colonna

# stratify_col2 = stratify_col[id_push_temp]

# id_push,id_valid = train_test_split(id_push_temp,random_state=27, test_size=0.20, stratify=stratify_col2)




path_dst_push = os.path.join(dest_dir,'push')
path_dst_valid = os.path.join(dest_dir,'valid')

# if not os.path.exists(path_dst_push):
#     os.makedirs(path_dst_push)
# if not os.path.exists(path_dst_valid):
#     os.makedirs(path_dst_valid)
    
# for a in [path_dst_push, path_dst_valid]:
#     for x in ['mass', 'calc']:
#         for y in ['benign', 'malignant', 'no_callback']:
#             path = os.path.join(a, x, y)
#             if not os.path.exists(path):
#                 os.makedirs(path)

# for a in [path_dst_push, path_dst_valid]:
#     for x in ['benign', 'malignant']:
#         for y in ['mass', 'calc']:
#             path = os.path.join(a, x, y)
#             if not os.path.exists(path):
#                 os.makedirs(path)

#TODO 7 aprile 2022
for a in [path_dst_push, path_dst_valid]:
    for x in ['benign', 'cancer']:
        path = os.path.join(a, x)
        if not os.path.exists(path):
            os.makedirs(path)

if corrupted_png_dir is not None:
    print('')
    # # id_push_orgn,id_push_crrp = train_test_split(id_push,random_state=27, test_size=0.5)
    # # id_test_orgn,id_test_crrp = train_test_split(id_test,random_state=27, test_size=0.5)
    # # id_valid_orgn,id_valid_crrp = train_test_split(id_valid,random_state=27, test_size=0.5)
    # number_of_corruptions = len(glob.glob(os.path.join(corrupted_png_dir,'*')))
    # tupla_push = np.array_split(push_names, np.round(number_of_corruptions+1,decimals=1)) # e.g., tupla =(id_push_original, id_push_corrupted1, id_pushcorrupted2, [...])
    # tupla_valid = np.array_split(valid_names, np.round(number_of_corruptions+1,decimals=1))

    # corr_subdir = glob.glob(os.path.join(corrupted_png_dir, '*'))
    
    # file_names_push=list()
    # file_lab_s_push=list() # label di severità HG o LG (Malignant Benign)
    # file_lab_c_push=list() # label di corruzione 1 o 0
    # laterality_push = list()
    # for k, list_corr in enumerate(tupla_push):
    #     for name in list_corr:
    #         name = str(name)
    #         laterality = df.loc[name,'LeftRight']
    #         laterality_push.append(laterality)
    #         if k == 0: # k == 0 cioè le immagini non corrotte
    #             shutil.copy(os.path.join(png_dir, name), os.path.join(path_dst_push, name))
    #             file_names_push.append(name)
    #             file_lab_c_push.append(0)
    #         else:
    #             a = glob.glob(os.path.join(corr_subdir[k-1], '*'+name))
    #             a = str(a[0])
    #             file_name = os.path.basename(a)
    #             file_names_push.append(file_name)   
    #             file_lab_c_push.append(1)
    #             shutil.copy(os.path.join(corr_subdir[k-1], file_name), os.path.join(path_dst_push, file_name))
    #         file_lab_s_push.append(df.loc[name,'Label'])

    # df_push = pd.DataFrame(
    #     data ={'File name':file_names_push,
    #            'LeftRight': laterality_push,
    #            'Label severity': file_lab_s_push,
    #            'Label corruption': file_lab_c_push},                
    #     )
    
    
    # file_names_test=list()
    # file_lab_s_test=list() # label di severità HG o LG
    # file_lab_c_test=list() # label di corruzione 1 o 0
    # laterality_test = list()
    # for k, list_corr in enumerate(tupla_test):
    #     for name in list_corr:
    #         name=str(name)
    #         laterality = df.loc[name,'LeftRight']
    #         laterality_test.append(laterality)
    #         if k == 0:
    #             shutil.copy(os.path.join(png_dir, name), os.path.join(path_dst_test, name))
    #             file_names_test.append(name)
    #             file_lab_c_test.append(0)
    #         else:
    #             a = glob.glob(os.path.join(corr_subdir[k-1], '*'+name))
    #             a = str(a[0])
    #             file_name = os.path.basename(a)
    #             file_names_test.append(file_name)   
    #             file_lab_c_test.append(1)
    #             shutil.copy(os.path.join(corr_subdir[k-1], file_name), os.path.join(path_dst_test, file_name))
    #         file_lab_s_test.append(df.loc[name,'Label'])

    # df_test = pd.DataFrame(
    #     data ={'File name':file_names_test,
    #            'LeftRight': laterality_test,
    #            'Label severity': file_lab_s_test,
    #            'Label corruption': file_lab_c_test},                
    #     )
    
    
    
    
    # file_names_valid=list()
    # file_lab_s_valid=list() # label di severità HG o LG
    # file_lab_c_valid=list() # label di corruzione 1 o 0  
    # laterality_valid = list()
    # for k, list_corr in enumerate(tupla_valid):
    #     for name in list_corr:
    #         name=str(name)
    #         laterality = df.loc[name,'LeftRight']
    #         laterality_valid.append(laterality)
    #         if k == 0:
    #             shutil.copy(os.path.join(png_dir, name), os.path.join(path_dst_valid, name))
    #             file_names_valid.append(name)
    #             file_lab_c_valid.append(0)
    #         else:
    #             a = glob.glob(os.path.join(corr_subdir[k-1], '*'+name))
    #             a= str(a[0])
    #             file_name = os.path.basename(a)
    #             file_names_valid.append(file_name)   
    #             file_lab_c_valid.append(1)
    #             shutil.copy(os.path.join(corr_subdir[k-1], file_name), os.path.join(path_dst_valid, file_name))

    #         file_lab_s_valid.append(df.loc[name,'Label'])

    # df_valid = pd.DataFrame(
    #     data ={'File name':file_names_valid,
    #            'LeftRight': laterality_valid,
    #            'Label severity': file_lab_s_valid,
    #            'Label corruption': file_lab_c_valid},                
    #     )



else:
         
    df_push = df.loc[push_names,('label')]    
    df_valid = df.loc[valid_names,('label')]
    
    for name in push_names:
        label_name = df.at[name,'label']
        # TODO: le prossime righe sono per il task benigno vs maligno
        # name_old = name
        # splits = name.split(sep='/')
        # name = os.path.join(splits[1], splits[0], splits[2])
        
        #
        shutil.copy(os.path.join(png_dir,name),os.path.join(path_dst_push,label_name,os.path.basename(name)))
           
    for name in valid_names:
        # # TODO: le prossime righe sono per il task benigno vs maligno
        # name_old = name
        # splits = name.split(sep='/')
        # name = os.path.join(splits[1], splits[0], splits[2])
        # shutil.copy(os.path.join(png_dir,name_old),os.path.join(path_dst_valid,name))
        
        #
        label_name = df.at[name,'label']
        shutil.copy(os.path.join(png_dir,name),os.path.join(path_dst_valid,label_name,os.path.basename(name)))


# if corrupted_png_dir is not None:
#     df_push.to_csv(os.path.join(path_dst_push,'labels_push.csv'),sep=',',index=False)
#     df_valid.to_csv(os.path.join(path_dst_valid,'labels_valid.csv'),sep=',',index=False)
# else:
df_push.to_csv(os.path.join(dest_dir,'labels_push.csv'),sep=',',index=True)
df_valid.to_csv(os.path.join(dest_dir,'labels_valid.csv'),sep=',',index=True)

print('Done.')