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
    parser = argparse.ArgumentParser(description='Plot the resuts of yolov5 with the original labels and yolov5 with the same labels on all of the slices, together with the ground truth.')
    parser.add_argument('yolov5_dir', help='Absolute path of the folder containing the images resulting from yolov5 inference, with the original labels')
    parser.add_argument('yolov5_SL_dir', help='Absolute path of the folder containing the images resulting from yolov5 inference, with same labels on all the slices')
    parser.add_argument('gt_dir', help='Absolute path to folder of input images (parent of images/ and labels/)')
    parser.add_argument('out_dir', help='Absolute path to folder where the plots will be saved')
    args = parser.parse_args()

    yolov5_dir = args.yolov5_dir
    yolov8_dir = args.yolov5_SL_dir
    gt_dir = args.gt_dir
    out_dir = args.out_dir

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    path_to_imgs = os.path.join(gt_dir, 'images')
    path_to_labels = os.path.join(gt_dir, 'labels')

    imgs_list = glob.glob(os.path.join(path_to_imgs, '*'))

    for img in imgs_list:
        # scorro su tutte le immagini, per ciascuna prendo la GT e la plotto insieme a yv5 e yv8
        name = os.path.basename(img)
        name = name.split('.')[0]

        label = os.path.join(path_to_labels, f'{name}.txt')
        img_pil = Image.open(img)
        img_npy = np.array(img_pil)
        img_width = img_npy.shape[1]
        img_height = img_npy.shape[0]

        yolov5_img = os.path.join(yolov5_dir, f'{name}.png')
        yolov8_img = os.path.join(yolov8_dir, f'{name}.png')

        # Specify the font and size
        #font = ImageFont.truetype("Supplemental/Futura.ttc", 128)
        font = ImageFont.truetype("URWGothic-Book.otf", 128)
        fill = (153, 153, 255)
        font2 = ImageFont.truetype("URWBookman-Light.otf", 64)

        #fill = (153, 153, 255)
        #font2 = ImageFont.truetype("Supplemental/Futura.ttc", 64)


        yolov5_pil = Image.open(yolov5_img)
        # yolov5_pil = ImageOps.grayscale(yolov5_pil)
        # yolov5_npy = np.array(yolov5_pil)

        # Create an ImageDraw object
        draw_v5 = ImageDraw.Draw(yolov5_pil)
        # Draw the text on the image at a given position and color
        draw_v5.text((500, 100), "Yolo v5 Orig. Label", font=font, fill=fill)
        draw_v5.text((500, 250), f"{name}", font=font2, fill=fill)


        yolov8_pil = Image.open(yolov8_img)
        # yolov8_pil = ImageOps.grayscale(yolov8_pil)
        # yolov8_npy = np.array(yolov8_pil)

        # Create an ImageDraw object
        draw_v8 = ImageDraw.Draw(yolov8_pil)
        # Draw the text on the image at a given position and color
        draw_v8.text((500, 100), "Yolo v5 Same Labels", font=font, fill=fill)



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
        
        # le dimensioni delle immagini coincidono? reshape a 1280? i canali sono 3 per tutte le immagini?
            # cmap='gray'? Altrimenti vanno aperte in gray con PIL

        # fig, (ax1, ax2, ax3) = plt.subplots(1, 3)
        # fig.suptitle(f'{name}')
        # ax1.imshow(yolov5_npy, cmap='gray')
        # ax1.set_title('Yolo v5')
        # ax2.imshow(img_npy, cmap='gray')
        # ax2.set_title('Ground truth')
        # # ax3.imshow(yolov8_npy, cmap='gray')
        # ax3.imshow(yolov8_npy)
        # ax3.set_title('Yolo v8')
        
        img_gt_pil = Image.fromarray(img_npy)
        img_gt_rgb = img_gt_pil.convert('RGB')

        # Create an ImageDraw object
        draw_gt = ImageDraw.Draw(img_gt_rgb)
        # Draw the text on the image at a given position and color
        draw_gt.text((500, 100), "Ground truth", font=font, fill=fill)


        # img_modified_pil.show()
        # yolov5_pil.show()
        # yolov8_pil.show()

        # Get the width and height of each image
        width_gt, height_gt = img_gt_rgb.size
        width_v5, height_v5 = yolov5_pil.size
        width_v8, height_v8 = yolov8_pil.size

        # Create a new image with the same height as the images and the sum of their widths
        new_width = width_v5 + width_gt + width_v8
        new_height = max(height_v5, height_gt, height_v8)
        new_image = Image.new("RGB", (new_width, new_height))

        # Paste the images side by side on the new image
        new_image.paste(yolov5_pil, (0, 0))
        new_image.paste(img_gt_rgb, (width_v5, 0))
        new_image.paste(yolov8_pil, (width_v5 + width_gt, 0))

        # salvataggio immagini
        # out_name = os.path.join(out_dir, f'{name}.pdf')
        # plt.savefig(out_name, bbox_inches='tight')

        out_name = os.path.join(out_dir, f'{name}.png')
        new_image.save(out_name)
        # plt.savefig(out_name, dpi=500, bbox_inches='tight')




if __name__ == "__main__":
    main()
