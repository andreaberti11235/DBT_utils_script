import argparse
from PIL import Image
from ultralytics import YOLO
import os
from glob import glob

parse = argparse.ArgumentParser(description="Script to ensemble two yolo models and perform inference.")
parse.add_argument('v5', help='Path to the yolo v5 model', type=str)
parse.add_argument('v8', help='Path to the yolo v8 model', type=str)
parse.add_argument('path_to_imgs', help='Path to the images folder', type=str)
parse.add_argument('path_to_output', help='Path to the yolo v8 model', type=str)

args = parse.parse_args()

v5_path = args.v5
v8_path = args.v8
path_to_imgs = args.path_to_imgs
path_to_output = args.path_to_output

# Load the YOLOv5 and YOLOv8 models from their weights
model5 = ultralytics.YOLO(v5_path)
model8 = ultralytics.YOLO(v8_path)

# Create an ensemble of the two models
model = ultralytics.Ensemble([model5, model8])

if not os.path.exists(path_to_output):
    os.makedirs(path_to_output)

img_list = glob(os.path.join(path_to_imgs, '*'))

for img in img_list:
    model.predict(img, save=True, imgsz=1280, iou=0.1, conf=0.1)