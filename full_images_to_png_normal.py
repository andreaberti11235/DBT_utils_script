import os
import argparse
import cv2
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pydicom
from PIL import Image
import scipy.ndimage
import shutil
from tqdm import tqdm

def draw_box(
    image,
    x,
    y,
    width,
    height,
    color = None,
    lw=4,
):
    
    """Draw bounding box on an image

    Arguments:
        image {np.ndarray}: Numpy array of the image
        x {int}: X coordinate of the bounding-box (from csv)
        y {int}: Y coordinate of the bounding-box (from csv)
        width {int}: Width of the bounding-box (from csv)
        hight {int}: Hight of the bounding-box (from csv)
        color {}: Color of the bounding-box (default: {None})
        lw {int}: Line width (default: {4})

    Returns:
        image {np.ndarray}: Numpy array of the image with the bounding-box
    """

    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)
    if color is None:
        color = np.max(image)
    if len(image.shape) > 2 and not hasattr(color, "__len__"):
        color = (color,) + (0,) * (image.shape[-1] - 1)
    image[y : y + lw, x : x + width] = color
    image[y + height - lw : y + height, x : x + width] = color
    image[y : y + height, x : x + lw] = color
    image[y : y + height, x + width - lw : x + width] = color
    return image

def crop_mass(image,
    x,
    y,
    width,
    height,
):
    
    """Crop mass given the bounding box on an image

    Arguments:
        image {np.ndarray}: Numpy array of the image
        x {int}: X coordinate of the bounding-box (from csv)
        y {int}: Y coordinate of the bounding-box (from csv)
        width {int}: Width of the bounding-box (from csv)
        hight {int}: Hight of the bounding-box (from csv)

    Returns:
        mass_cropped {np.ndarray}: Numpy array of the cropped mass
    """
    x = min(max(x, 0), image.shape[1] - 1)
    y = min(max(y, 0), image.shape[0] - 1)

    mass_cropped = image[y : y + height, x : x + width]
    return mass_cropped

def _get_image_laterality(pixel_array: np.ndarray) -> str:
    left_edge = np.sum(pixel_array[:, :, 0])  # sum of left edge pixels
    right_edge = np.sum(pixel_array[:, :, -1])  # sum of right edge pixels
    return "R" if left_edge < right_edge else "L"

def mass_slice_and_create_pil(npy_img, slice, apply_clahe):
    mass_slice = npy_img[slice, :, :]
    mass_slice = ((mass_slice - np.amin(mass_slice))/(np.amax(mass_slice) - np.amin(mass_slice)))*255
    mass_slice = mass_slice.astype(np.uint8)

    if apply_clahe:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        clahe_img = clahe.apply(mass_slice)
        clahe_img = ((clahe_img - np.amin(clahe_img))/(np.amax(clahe_img) - np.amin(clahe_img)))*255
        clahe_img = clahe_img.astype(np.uint8)
        pil_mass = Image.fromarray(clahe_img)
    else:    
        pil_mass = Image.fromarray(mass_slice)
    return pil_mass


def find_selected_slices(num_slices_each_side, slice, volume_slices, label):
    if label == 'cancer':
        num_slices_each_side += 1
    if volume_slices / 4 <= 4 * num_slices_each_side: # prenderei le slice a step di due se ce ne fossero abbastanza
        step = 1
        start = max(0, slice - num_slices_each_side)
            # selected_slices = np.arange(slice - num_slices_each_side, slice + num_slices_each_side + 1)
    else:
        step = 2
        start = max(0, slice - 2 * num_slices_each_side)
    selected_slices = np.arange(0, 2 * num_slices_each_side + 1) * step + start    
    dim = np.size(selected_slices)
    # selected_slices = selected_slices[selected_slices != slice]
    if np.size(selected_slices) == dim:
            # se la prima slice risulta essere negativa, allora partiamo da 0, se però la slice annotata non cade 
            # tra quelle selezionate (e quindi rimaniamo con 1 slice in più), eliminiamo l'ultima slice
            # si può fare meglio?
        selected_slices = selected_slices[:-1]
    return selected_slices

def find_z_intervals(central_slice, volume_slices):
    mass_depth = np.round(volume_slices / 4)
    first_edge = max(0, central_slice - (mass_depth // 2))
    second_edge = max(mass_depth, first_edge + mass_depth)
    return first_edge, second_edge

def overlap(interval1, interval2):
    """
    Given [0, 4] and [1, 10] returns [1, 4]
    """
    if interval2[0] <= interval1[0] <= interval2[1]:
        start = interval1[0]
    elif interval1[0] <= interval2[0] <= interval1[1]:
        start = interval2[0]
    else:
        start = 0

    if interval2[0] <= interval1[1] <= interval2[1]:
        end = interval1[1]
    elif interval1[0] <= interval2[1] <= interval1[1]:
        end = interval2[1]
    else:
        end = 0

    return (start, end)

def main():
    parser = argparse.ArgumentParser(description='Crop masses and save as png.')
    parser.add_argument('dest_dir', help='Destination directory of images (absolute path)')
    parser.add_argument('path_to_imgs', help='Absolute path to folder of input images (parent of normal)')
    parser.add_argument('-c', '--clahe', help='Apply CLAHE preprocessing to images', default=False, action='store_true')
    args = parser.parse_args()

    dest_dir = args.dest_dir
    path_to_imgs = args.path_to_imgs
    apply_clahe = args.clahe
    
    dest_dir = os.path.join(dest_dir, 'normal')
    path_to_imgs = os.path.join(path_to_imgs, 'normal')


    for idx, image in tqdm(enumerate(os.listdir(path_to_imgs))):
        
        img_path = os.path.join(path_to_imgs, image)
        extension = image.split(sep='.')[-1]
        
        if extension == 'dcm':

            # opening the dicom file and converting it ot numpy
            dcm_img = pydicom.dcmread(img_path)
            npy_img = dcm_img.pixel_array
            # img_height = npy_img.shape[1]
            # img_width = npy_img.shape[2]
            img_depth = npy_img.shape[0]

            central_slice = img_depth // 2
            slice_number = central_slice
            # creating the PIL file of the annotated slice
            pil_mass = mass_slice_and_create_pil(npy_img, slice_number, apply_clahe=apply_clahe)
            # getting the name without the dcm extension
            original_file_name = image.split(sep='.')[0]

            # creating the name of the image png file and of the corresponding txt file for the label. They
            # have the same name and different extension
            out_img_name = f'{original_file_name}_slice{slice_number}.png'
            out_txt_label = f'{original_file_name}_slice{slice_number}.txt'
            
            # in dest_dir, we will have images/ with the png files and labels/ with the txt files
            out_dir_path_imgs = os.path.join(dest_dir, 'original', 'images')
            out_path = os.path.join(out_dir_path_imgs, out_img_name)

            if not os.path.exists(out_dir_path_imgs):
                os.makedirs(out_dir_path_imgs)

            # creare anche la dir per le label
            out_dir_path_labels = os.path.join(dest_dir, 'original', 'labels')
            out_path_label = os.path.join(out_dir_path_labels, out_txt_label)

            if not os.path.exists(out_dir_path_labels):
                os.makedirs(out_dir_path_labels)

    

            # otherwise, create the txt file for the label and save the png file
            with open(out_path_label, "w") as label_file:
                label_file.write(f'')

            pil_mass.save(out_path, 'PNG')
           

# c'è un problema di fondo con le slice centrali, infatti non stanno nella lista delle selected slice, in questo modo non vengono mai controllate nelle 
# varie ciclate e non vengono aggiunte le slice di eventuali altre masse. Possibile soluzione: aggiungo l'indice di slice anche lì e aggiungo quell'indice alle selected slices, ma
# devo stare attento nelle varie ciclate a non creare due volte la stessa slice (ad es. mettere un check if selected_slice != central_slice)

if __name__ == "__main__":
    main()