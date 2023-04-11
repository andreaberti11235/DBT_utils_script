import os
import argparse
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import pydicom
from PIL import Image
import scipy.ndimage
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



parser = argparse.ArgumentParser(description='Crop masses and save as png.')
parser.add_argument('csv_box_file', help='csv file of bounding boxes (absolute path)')
parser.add_argument('dest_dir', help='Destination directory of images (absolute path)')
parser.add_argument('path_to_imgs', help='Absolute path to folder of input images (parent of Breast-Cancer-Screening-DBT)')
args = parser.parse_args()

csv_path = args.csv_box_file
dest_dir = args.dest_dir
path_to_imgs = args.path_to_imgs
df_box = pd.read_csv(csv_path)

for idx, row in tqdm(df_box.iterrows(), total=len(df_box.index)):
    # extract info from the csv
    label = row['Class']
    patient_id = row['PatientID']
    study_id = row['StudyUID']
    view = row['View']

    slice = row['Slice']
    x = row['X']
    y = row['Y']
    width = row['Width']
    height = row['Height']
    volume_slices = row['VolumeSlices']

    img_name = f'{patient_id}_{study_id}_{view}.dcm'
    img_path = os.path.join(path_to_imgs, label, img_name)

    # opening the dicom file and converting it ot numpy
    dcm_img = pydicom.dcmread(img_path)
    dcm_img = pydicom.dcmread(img_path)
    npy_img = dcm_img.pixel_array
    # some images have to be reoriented
    view_laterality = view[0].upper()
    image_laterality = _get_image_laterality(npy_img)
    if not image_laterality == view_laterality:
            npy_img = np.flip(npy_img, axis=(-1, -2))

    # img_bbox = np.copy(npy_img)
    # img_bbox = draw_box(image=img_bbox[slice, :, :], x=x, y=y, width=width, height=height, lw=10)
    

    mass_cropped = crop_mass(image=npy_img[slice, :, :], x=x, y=y, width=width, height=height)
    # plt.subplot(121)
    # plt.imshow(img_bbox, cmap='gray')
    # plt.subplot(122)
    # plt.imshow(mass_cropped, cmap='gray')
    # plt.show()

    # riscalare le immagini crop a max e min del crop?
    mass_cropped = ((mass_cropped - np.amin(mass_cropped))/(np.amax(mass_cropped) - np.amin(mass_cropped)))*255
    mass_cropped =mass_cropped.astype(np.uint8)

    # create PIL image and save as png
    pil_mass = Image.fromarray(mass_cropped)
    out_file_name = img_name.split(sep='.')[0]
    out_file_name = f'{out_file_name}.png'

    out_path = os.path.join(dest_dir, label, out_file_name)

    # aggiungere sottocartella 'original', in modo che sia .../original/benign, e un'altra con un altro nome per distinguere le slice annotate da quelle che prendiamo in più
    pil_mass.save(out_path, 'PNG')