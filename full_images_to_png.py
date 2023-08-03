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

def idx_square_box(idx, og_size, new_size):
    center = idx + (og_size // 2) 
    idx_new = center - (new_size // 2)
    return idx_new

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
    parser.add_argument('csv_box_file', help='csv file of bounding boxes (absolute path)')
    parser.add_argument('dest_dir', help='Destination directory of images (absolute path)')
    parser.add_argument('path_to_imgs', help='Absolute path to folder of input images (parent of benign/cancer)')
    parser.add_argument('-n', '--num_slices', type=int, default=3, help='Number (for each side) of additional slices next to the central slice qith lesion (default 3)')
    parser.add_argument('-c', '--clahe', help='Apply CLAHE preprocessing to images', default=False, action='store_true')
    args = parser.parse_args()

    csv_path = args.csv_box_file
    dest_dir = args.dest_dir
    path_to_imgs = args.path_to_imgs
    num_slices_each_side = args.num_slices
    apply_clahe = args.clahe

    df_box = pd.read_csv(csv_path)

    mass_count = 0
    img_name_old = ''
    selected_slices_previous_mass = []
    z_interval_previous_mass = []
    central_slices_previous_masses = []
    bbox_previous_mass = []

    for idx, row in tqdm(df_box.iterrows(), total=len(df_box.index)):
        # extract info from the csv
        label = row['Class']
        patient_id = row['PatientID']
        study_id = row['StudyUID']
        view = row['View']

        slice_number = row['Slice']
        x = row['X']
        y = row['Y']
        width = row['Width']
        height = row['Height']
        volume_slices = row['VolumeSlices']
        
        img_name = f'{patient_id}_{study_id}_{view}.dcm'
        img_path = os.path.join(path_to_imgs, label, img_name)

        x_center = x + width // 2
        y_center = y + height // 2

        # check if same patient dcm
        if img_name == img_name_old:
            mass_count += 1
            # print(f'Same mass in row {idx}')
        else:
            mass_count = 0
            selected_slices_previous_mass = []
            z_interval_previous_mass = []
            central_slices_previous_masses = []
            bbox_previous_mass = []


        img_name_old = img_name

        # opening the dicom file and converting it ot numpy
        dcm_img = pydicom.dcmread(img_path)
        npy_img = dcm_img.pixel_array
        # some images have to be reoriented
        view_laterality = view[0].upper()
        image_laterality = _get_image_laterality(npy_img)
        if not image_laterality == view_laterality:
                npy_img = np.flip(npy_img, axis=(-1, -2))

        # creating the PIL file of the annotated slice
        pil_mass = mass_slice_and_create_pil(npy_img, slice_number, apply_clahe=apply_clahe)
        # getting the name without the dcm extension
        original_file_name = img_name.split(sep='.')[0]

        # creating the name of the image png file and of the corresponding txt file for the label. They
        # have the same name and different extension
        out_img_name = f'{original_file_name}_mass{mass_count}_slice{slice_number}.png'
        out_txt_label = f'{original_file_name}_mass{mass_count}_slice{slice_number}.txt'
        
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

        # creating the folder for the augmented images (augmented/images/) and their labels (augmented/labels/)
        out_dir_path_imgs_augm = os.path.join(dest_dir, 'augmented', 'images')
        out_path_augm = os.path.join(out_dir_path_imgs_augm, out_img_name)

        if not os.path.exists(out_dir_path_imgs_augm):
            os.makedirs(out_dir_path_imgs_augm)

        out_dir_path_label_augm = os.path.join(dest_dir, 'augmented', 'labels')
        out_path_label_augm = os.path.join(out_dir_path_label_augm, out_txt_label)

        if not os.path.exists(out_dir_path_label_augm):
            os.makedirs(out_dir_path_label_augm)

        # fare un check se coincide con la stessa slice centrale di masse precedenti dello stesso paziente
        if slice_number in central_slices_previous_masses:
            # if mass in the same slice of a previous one, update the label file of that mass
            mass_nr = central_slices_previous_masses.index(slice_number) # this returns the position of the first occurrance of the value
            previous_label_path = os.path.join(out_dir_path_labels, f'{original_file_name}_mass{mass_nr}_slice{slice_number}.txt')
            string_to_be_written = f'0, {x_center}, {y_center}, {width}, {height}'
            with open(previous_label_path, 'a') as previous_label:
                previous_label.write(f'\n{string_to_be_written}')
            
            print(f'Mass {mass_count} was in the same slice as mass {mass_nr}')

            # # also update the label in the augmented folder
            # previous_label_path_augm = os.path.join(out_dir_path_label_augm, f'{original_file_name}_mass{mass_nr}_slice{slice_number}.txt')
            # with open(previous_label_path_augm, 'a') as previous_label:
            #     previous_label.write(f'\n{string_to_be_written}')
                
        else:
            # otherwise, create the txt file for the label and save the png file
            with open(out_path_label, "w") as label_file:
                label_file.write(f'0, {x_center}, {y_center}, {width}, {height}')

            pil_mass.save(out_path, 'PNG')
            # ricordarsi di aggiungere CLAHE (forse si può mettere anche dentro alla funzione mass_slice_and_create_pil)

            # # making the exact same copy of the original and label image in the augmented folder
            # pil_mass.save(out_path_augm, 'PNG')
            # shutil.copy(out_path_label, out_path_label_augm)


        # volume_slices is the depth of the whole image, in the wiki-page it is stated that
        # the mass can be considered to extend for 25% of the whole volume.
        # When possible, we take 3 slices for each side, with a step of 2, otherwise, we take them contiguously (see function)
        selected_slices = find_selected_slices(num_slices_each_side, slice_number, volume_slices, label=label)
        z_interval = find_z_intervals(slice_number, volume_slices)

        # initialising a dictionary containing the slices of the new mass that overlap with one of previous masses, 
        # but were not selected in the previous iterations

        # if it is not the first mass, check if there is overlap in slices with the previous masses (if same patient)
        if mass_count != 0:
            for i, previous_interval in enumerate(z_interval_previous_mass):
                overlap_interval = overlap(z_interval, previous_interval)
                if overlap_interval != (0, 0):
                    # if the masses appear on the same slices, check if those are selected slices
                    for old_slice in selected_slices_previous_mass[i]:
                        # for new_slice in selected_slices:
                            if overlap_interval[0] <= old_slice <= overlap_interval[1]:
                            # if old_slice == new_slice:
                                # questa cosa non è corretta, non devo solo vedere quando sono uguali le slice, ma semplicemente quando stanno nell'intervallo, un po' come ho 
                                # fatto sotto, stessa cosa
                                print(f'New mass {mass_count} overlaps with previous mass {i} in slice {old_slice} for patient {original_file_name}. First check.')
                                # adding the bbox of the new mass to the label file 
                                file_name = f'{original_file_name}_mass{i}_slice{old_slice}.txt'
                                file_path = os.path.join(out_dir_path_label_augm, file_name)
                                string_to_be_written = f'0, {x_center}, {y_center}, {width}, {height}'
                                with open(file_path, 'a') as label_file_old:
                                    label_file_old.write(f'\n{string_to_be_written}')
                                
                    for new_slice in selected_slices:
                        if new_slice in selected_slices_previous_mass[i]:
                            print(f'Slice {new_slice} of mass {mass_count} already present in mass {i}. Removing mass from list.')
                            # removing the slice for the new mass
                            selected_slices = selected_slices[selected_slices != new_slice]

        for selected_slice in selected_slices:
            pil_mass_augm = mass_slice_and_create_pil(npy_img, selected_slice, apply_clahe=apply_clahe)
            # base_out_name = original_file_name.split(sep='.')[0]########### levare la slice e sostituirla con quella attuale, usare original_file_name?
            out_file_name_augm = f'{original_file_name}_mass{mass_count}_slice{selected_slice}.png'
            out_file_name_label_augm = f'{original_file_name}_mass{mass_count}_slice{selected_slice}.txt'
            out_slice_augm_path = os.path.join(out_dir_path_imgs_augm, out_file_name_augm)
            out_label_augm_path = os.path.join(out_dir_path_label_augm, out_file_name_label_augm)
            with open(out_label_augm_path, "w") as label_file:
                label_file.write(f'0, {x_center}, {y_center}, {width}, {height}')
            pil_mass_augm.save(out_slice_augm_path, 'PNG')

            if mass_count != 0:
                for i, previous_interval in enumerate(z_interval_previous_mass):
                    overlap_interval = overlap(z_interval, previous_interval)
                    if overlap_interval != (0, 0):
                        # if the masses appear on the same slices, check if those are selected slices
                        # for old_slice in selected_slices_previous_mass[i]:
                            # for new_slice in selected_slices:
                                if overlap_interval[0] <= selected_slice <= overlap_interval[1]:
                                    print(f'Previous mass ({i}) overlaps with current mass in slice {selected_slice}. Adding previous mass label to current mass label file. Triggered for {original_file_name} at mass {mass_count}. Second check.')
                                    # base_out_name = out_file_name.split(sep='.')[0]############
                                    # out_file_name_label_augm = f'{base_out_name}_slice{selected_slice}.txt'
                                    # out_label_augm_path = os.path.join(out_dir_path_label_augm, out_file_name_label_augm)
                                    x_center_old = bbox_previous_mass[i][0] # the tuple does not contain the label (0)
                                    y_center_old = bbox_previous_mass[i][1]
                                    width_old = bbox_previous_mass[i][2]
                                    height_old = bbox_previous_mass[i][3]
                                    string_to_be_written = f'0, {x_center_old}, {y_center_old}, {width_old}, {height_old}'
                                    with open(out_label_augm_path, "a") as label_file:
                                        label_file.write(f'\n{string_to_be_written}')
                                    # errore qui: penso che non serva il for alla riga 307, com'è ora per ogni slice selezionata della vecchia massa
                                    # scorro di nuovo tutte le slice della nuova che stanno nell'intervallo e guardo quali stanno nell'intervallo e modifico il file 
                                    # ma in questo modo modifico il file della label molte volte, infatti mi trovo i bbox della vecchia massa copiati molte volte per ogni file della nuova massa


        z_interval_previous_mass.append(z_interval)
        bbox_previous_mass.append((x_center, y_center, width, height))
        # appending the slice index to the list
        central_slices_previous_masses.append(slice_number)
        selected_slices_previous_mass.append(selected_slices)

# c'è un problema di fondo con le slice centrali, infatti non stanno nella lista delle selected slice, in questo modo non vengono mai controllate nelle 
# varie ciclate e non vengono aggiunte le slice di eventuali altre masse. Possibile soluzione: aggiungo l'indice di slice anche lì e aggiungo quell'indice alle selected slices, ma
# devo stare attento nelle varie ciclate a non creare due volte la stessa slice (ad es. mettere un check if selected_slice != central_slice)

if __name__ == "__main__":
    main()