import argparse
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageOps, ImageDraw, ImageFont


def draw_box(
    image,
    x_center,
    y_center,
    width,
    height,
    img_width,
    img_height,
    color = None,
    lw=4,
):
    
    """Draw bounding box on an image

    Arguments:
        image {np.ndarray}: Numpy array of the image
        x {int}: X coordinate of the bounding-box (from csv)
        y {int}: Y coordinate of the bounding-box (from csv)
        width {float}: Width of the bounding-box (from csv)
        hight {float}: Hight of the bounding-box (from csv)
        img_width {int}: Width of the NPY image
        img_height {int}: Height of the NPY image
        color {}: Color of the bounding-box (default: {None})
        lw {int}: Line width (default: {4})

    Returns:
        image {np.ndarray}: Numpy array of the image with the bounding-box
    """
    # converto
    x = int(x_center * img_width - (width * img_width) // 2)
    y = int(y_center * img_height - (height * img_height) // 2)

    width = int(width * img_width)
    height = int(height * img_height)

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

def main():
    parser = argparse.ArgumentParser(description='Plot the resuts of yolov5 and yolov8, together with the ground truth.')
    parser.add_argument('ensemble_dir', help='Absolute path of the folder containing the labels resulting from the ensemble inference')
    parser.add_argument('yolov5_dir', help='Absolute path of the folder containing the images resulting from yolov5 inference')
    parser.add_argument('yolov8_dir', help='Absolute path of the folder containing the images resulting from yolov8 inference')
    parser.add_argument('gt_dir', help='Absolute path to folder of input images (parent of images/ and labels/)')
    parser.add_argument('out_dir', help='Absolute path to folder where the plots will be saved')
    args = parser.parse_args()

    ensemble_dir = args.ensemble_dir
    yolov5_dir = args.yolov5_dir
    yolov8_dir = args.yolov8_dir
    gt_dir = args.gt_dir
    out_dir = args.out_dir

    offset = 3

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path_to_imgs = os.path.join(gt_dir, 'images')
    path_to_labels = os.path.join(gt_dir, 'labels')

    imgs_list = glob.glob(os.path.join(path_to_imgs, '*'))

    for img in imgs_list:
        # scorro su tutte le immagini, per ciascuna prendo la GT e la plotto insieme a yv5 e yv8
        name = os.path.basename(img)
        name = name.split('.')[0]
        print(f'image {name}')


        label = os.path.join(path_to_labels, f'{name}.txt')
        img_pil = Image.open(img)
        img_npy = np.array(img_pil)
        ensemble_npy = np.copy(img_npy)
        img_width = img_npy.shape[1]
        img_height = img_npy.shape[0]

        ensemble_label = os.path.join(ensemble_dir, f'{name}.txt')

        yolov5_img = os.path.join(yolov5_dir, f'{name}.png')
        yolov8_img = os.path.join(yolov8_dir, f'{name}.png')

        # Specify the font and size
        #font = ImageFont.truetype("Supplemental/Futura.ttc", 128)
        font = ImageFont.truetype("DejaVuSans.ttf", 128)
        # font = ImageFont.truetype("URWGothic-Book.otf", 128)
        fill = (153, 153, 255)
        # font2 = ImageFont.truetype("URWBookman-Light.otf", 64)
        font2 = ImageFont.truetype("DejaVuSans-ExtraLight.ttf", 64)


        #fill = (153, 153, 255)
        #font2 = ImageFont.truetype("Supplemental/Futura.ttc", 64)

        ensemble_df = pd.read_csv(ensemble_label, sep=' ', header=None)
        for idx in ensemble_df.index:
            # scorro su tutte le bbox e prendo i vari valori
            x_center = ensemble_df.iloc[idx][1]
            y_center = ensemble_df.iloc[idx][2]
            width = ensemble_df.iloc[idx][3]
            height = ensemble_df.iloc[idx][4]

            # modifico l'immagine 'disegnandoci' dentro le bbox
            ensemble_npy = draw_box(ensemble_npy, x_center=x_center, y_center=y_center, 
                     width=width, height=height, 
                     img_width=img_width, img_height=img_height, color=255)


        ensemble_pil = Image.fromarray(ensemble_npy)
        ensemble_rgb = ensemble_pil.convert('RGB')

        # Create an ImageDraw object
        draw_ensemble = ImageDraw.Draw(ensemble_rgb)
        # Draw the text on the image at a given position and color
        draw_ensemble.text((500, 100), "Yolo ensemble", font=font, fill=fill)
        draw_ensemble.text((500, 250), f"{name}", font=font2, fill=fill)

        # add the text with the conf value for each detection
        for idx in ensemble_df.index:
            x_center = ensemble_df.iloc[idx][1]
            y_center = ensemble_df.iloc[idx][2]
            width = ensemble_df.iloc[idx][3]
            height = ensemble_df.iloc[idx][4]
            confidence = ensemble_df.iloc[idx][5]

            x_position = x_center - width/2
            y_position = y_center - height/2

            draw_ensemble.text((x_position, y_position - offset), f"{np.round(confidence, decimals=2)}", font=font2, fill=fill)

        yolov5_pil = Image.open(yolov5_img)

        # Create an ImageDraw object
        draw_v5 = ImageDraw.Draw(yolov5_pil)
        # Draw the text on the image at a given position and color
        draw_v5.text((500, 100), "Yolo v5", font=font, fill=fill)
        draw_v5.text((500, 250), f"{name}", font=font2, fill=fill)


        yolov8_pil = Image.open(yolov8_img)

        # Create an ImageDraw object
        draw_v8 = ImageDraw.Draw(yolov8_pil)
        # Draw the text on the image at a given position and color
        draw_v8.text((500, 100), "Yolo v8", font=font, fill=fill)



        df = pd.read_csv(label, sep=' ', header=None)
        for idx in df.index:
            # scorro su tutte le bbox e prendo i vari valori
            x_center = df.iloc[idx][1]
            y_center = df.iloc[idx][2]
            width = df.iloc[idx][3]
            height = df.iloc[idx][4]

            # modifico l'immagine 'disegnandoci' dentro le bbox
            img_npy = draw_box(img_npy, x_center=x_center, y_center=y_center, 
                     width=width, height=height, 
                     img_width=img_width, img_height=img_height, color=255)
        
        img_gt_pil = Image.fromarray(img_npy)
        img_gt_rgb = img_gt_pil.convert('RGB')

        # Create an ImageDraw object
        draw_gt = ImageDraw.Draw(img_gt_rgb)
        # Draw the text on the image at a given position and color
        draw_gt.text((500, 100), "Ground truth", font=font, fill=fill)

        # Get the width and height of each image
        width_gt, height_gt = img_gt_rgb.size
        width_ensemble, height_ensemble = ensemble_pil.size
        width_v5, height_v5 = yolov5_pil.size
        width_v8, height_v8 = yolov8_pil.size

        # Create a new image with the same height as the images and the sum of their widths
        new_width = width_v5 + width_gt + width_v8 + width_ensemble
        new_height = max(height_v5, height_gt, height_v8, height_ensemble)
        new_image = Image.new("RGB", (new_width, new_height))

        # Paste the images side by side on the new image
        new_image.paste(yolov5_pil, (0, 0))
        new_image.paste(img_gt_rgb, (width_v5, 0))
        new_image.paste(yolov8_pil, (width_v5 + width_gt, 0))
        new_image.paste(ensemble_pil, (width_v5 + width_gt + width_v8, 0))

        # salvataggio immagini
        out_name = os.path.join(out_dir, f'{name}.png')
        new_image.save(out_name)
        # plt.savefig(out_name, dpi=500, bbox_inches='tight')



if __name__ == "__main__":
    main()
