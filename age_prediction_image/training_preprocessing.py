import random
import os

from imutils import paths
from shutil import copy

image_filtered_folder = "../resource/UTKFace_filtered/"
image_ge_21_folder = "../resource/UTKFace/"
image_paths = list(paths.list_images(image_ge_21_folder))
num_image_le_21 = 5223
num_image_ge_21 = 18485
# There are much more faces that larger than age 21, so we need to balance the training set.
samples = set(random.sample(range(0, 18485), num_image_le_21))
for (i, imagePath) in enumerate(image_paths):
    if i not in samples:
        continue
    image_name = imagePath.split("/")[-1]
    # copy picture to target folder
    copy(imagePath, image_filtered_folder + image_name)

