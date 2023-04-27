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

def crop_mas_and_create_pil(npy_img, slice, x, y, width, height):
    mass_cropped = crop_mass(image=npy_img[slice, :, :], x=x, y=y, width=width, height=height)
    mass_cropped = ((mass_cropped - np.amin(mass_cropped))/(np.amax(mass_cropped) - np.amin(mass_cropped)))*255
    mass_cropped = mass_cropped.astype(np.uint8)
    pil_mass = Image.fromarray(mass_cropped)
    return pil_mass

def idx_square_box(idx, og_size, new_size):
    center = idx + (og_size // 2) 
    idx_new = center - (new_size // 2)
    return idx_new


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Crop masses and save as png.')
    parser.add_argument('csv_box_file', help='csv file of bounding boxes (absolute path)')
    parser.add_argument('dest_dir', help='Destination directory of images (absolute path)')
    parser.add_argument('path_to_imgs', help='Absolute path to folder of input images (parent of Breast-Cancer-Screening-DBT)')
    parser.add_argument('-n', '--num_slices', type=int, default=3, help='Number (for each side) of additional slices next to the central slice qith lesion')
    args = parser.parse_args()

    csv_path = args.csv_box_file
    dest_dir = args.dest_dir
    path_to_imgs = args.path_to_imgs
    num_slices_each_side = args.num_slices

    df_box = pd.read_csv(csv_path)

    mass_count = 0
    img_name_old = ''

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

        # check if same patient dcm
        if img_name == img_name_old:
            mass_count += 1
            # print(f'Same mass in row {idx}')
        else:
            mass_count = 0

        img_name_old = img_name

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
        

        # mass_cropped = crop_mass(image=npy_img[slice, :, :], x=x, y=y, width=width, height=height)
        # # plt.subplot(121)
        # # plt.imshow(img_bbox, cmap='gray')
        # # plt.subplot(122)
        # # plt.imshow(mass_cropped, cmap='gray')
        # # plt.show()

        # # riscalare le immagini crop a max e min del crop?
        # mass_cropped = ((mass_cropped - np.amin(mass_cropped))/(np.amax(mass_cropped) - np.amin(mass_cropped)))*255
        # mass_cropped = mass_cropped.astype(np.uint8)

        # # create PIL image and save as png
        # pil_mass = Image.fromarray(mass_cropped)

        # Identify max dimention to create squared crops
        max_dim = np.maximum(width, height)
        if width >= height:
            y = idx_square_box(y, height, max_dim)
        else:
            x = idx_square_box(x, width, max_dim)

        pil_mass = crop_mas_and_create_pil(npy_img, slice, x, y, max_dim, max_dim)
        out_file_name = img_name.split(sep='.')[0]
        out_file_name = f'{out_file_name}_mass{mass_count}.png'

        out_path = os.path.join(dest_dir, 'original', label, out_file_name)

        if not os.path.exists(out_path):
            os.makedirs(out_path)

        # aggiungere sottocartella 'original', in modo che sia .../original/benign, e un'altra con un altro nome per distinguere le slice annotate da quelle che prendiamo in più
        pil_mass.save(out_path, 'PNG')

        out_path_augm = os.path.join(dest_dir, 'augmented', label, out_file_name)

        if not os.path.exists(out_path_augm):
            os.makedirs(out_path_augm)

        pil_mass.save(out_path_augm, 'PNG')
        # volume_slices is the depth of the whole image, in the wiki-page it is stated that
        # the mass can be considered to extend for 25% of the whole volume.
        # When possible, we take 3 slices for each side, with a step of 2, otherwise, we take them contiguous
        if volume_slices / 4 <= 4 * num_slices_each_side:
            selected_slices = np.arange(slice - num_slices_each_side, slice + num_slices_each_side + 1)
            selected_slices = selected_slices[selected_slices != slice]
        else:
            if slice - 2 * num_slices_each_side < 0:
                raise Exception(f'Mass in {img_name} is to close to the border!')
            selected_slices = np.arange(slice - 2 * num_slices_each_side, slice + 2 * num_slices_each_side + 1, step=2)
            selected_slices = selected_slices[selected_slices != slice]

        for i, selected_slice in enumerate(selected_slices):
            pil_mass_augm = crop_mas_and_create_pil(npy_img, selected_slice, x, y, max_dim, max_dim)
            base_out_name = out_file_name.split(sep='.')[0]
            out_file_name_augm = f'{base_out_name}_slice{i}.png'
            pil_mass_augm.save(out_path_augm, 'PNG')

