import os
import argparse
import numpy as np
import pandas as pd
import shutil
from tqdm import tqdm

def check_clean_create(dir):
    if os.path.exists(dir):
        print(f'WARNING: {dir} already exists! Moving its contents into {dir}/old')
        dest_old = os.path.join(dir, 'old')
        items = os.listdir(dir)
        if 'old' in items:
            os.rename(dest_old, os.path.join(dir, 'old_old'))
        items = os.listdir(dir)
        os.makedirs(dest_old)
        for item in items:
            shutil.move(src=os.path.join(dir, item), 
                        dst=os.path.join(dest_old, item))
    else:
        os.makedirs(dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Retrieve benign and cancer files and moves them in the destination folder.')
    parser.add_argument('csv_labels_file', help='csv file of labels (normal, actionable, benign, cancer) of DBT images (absolute path)')
    parser.add_argument('csv_paths_file', help='csv file of DBT paths (absolute path)')
    parser.add_argument('dest_dir', help='Destination directory of images (absolute path)')
    parser.add_argument('path_to_imgs', help='Absolute path to folder of input images (parent of Breast-Cancer-Screening-DBT)')
    args = parser.parse_args()

    csv_labels_file = args.csv_labels_file
    csv_paths_file = args.csv_paths_file
    dest_dir = args.dest_dir
    path_to_imgs = args.path_to_imgs

    # Create subdir benign and cancer in dest_dir to store respective cases
    normal_dest_dir = os.path.join(dest_dir, 'normal')

    # Check if benign_/cancer_dest_dir already exist. If so, create subdir old and move all previous contents there
    check_clean_create(normal_dest_dir)

    df_labels = pd.read_csv(csv_labels_file)
    df_paths = pd.read_csv(csv_paths_file)

    count_normal = 0

    for idx, row in tqdm(df_labels.iterrows(), total=len(df_labels.index)):
        # extract info from the csv
        label_normal = row['Normal']
        patient_id = row['PatientID']
        study_id = row['StudyUID']
        view = row['View']
        if label_normal == 1:
            # take the path for that patient from the other csv (match of three conditions)
            dcm_path = df_paths[(df_paths['PatientID'] == patient_id) & (df_paths['StudyUID'] == study_id) & (df_paths['View'] == view)]['descriptive_path'].iloc[0]
            dcm_path = str(dcm_path)
            # dcm_path = dcm_path.replace(" ", r"\ ")
            # names in the csv dont match the real ones, fixing the names
            dcm_path = dcm_path.replace('000000-', '000000-NA-')

            # copy the patient's files in the correct folder. The name of the final file will be 
            #  patientID_studyUID_view.dcm
            src = os.path.join(path_to_imgs, dcm_path)
            dst = os.path.join(normal_dest_dir, f'{patient_id}_{study_id}_{view}.dcm')
            shutil.copy(src=src, dst=dst)
            count_normal += 1

            if count_normal == 1000:
                break
        
        else:
            pass
        

