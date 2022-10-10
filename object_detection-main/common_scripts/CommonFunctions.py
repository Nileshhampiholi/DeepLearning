import os
import random
import shutil
import tarfile
import zipfile
import cv2
import gip.gip_io.gip_image as gip
import numpy as np
import pip
import wget
from PIL import Image
from git import Repo
import numpy as np


def copy_files(source, destination):
    shutil.copy(source, destination)


def create_test_train_dataset(labels, images_path, test_path, train_path, splt_percentage):
    for label in labels:
        files = os.listdir(os.path.join(images_path, label))
        img_files = list(filter(lambda x: '.jpg' in x, files))
        train_files, test_files = partition_file_names(splt_percentage, img_files)
        for file_name in train_files:
            copy_files(os.path.join(images_path, label, file_name), train_path)
            copy_files(os.path.join(images_path, label, os.path.splitext(file_name)[0], '.xml'), train_files)
        for file_name in test_files:
            copy_files(os.path.join(images_path, label, file_name), test_path)
            copy_files(os.path.join(images_path, label, os.path.splitext(file_name)[0], '.xml'), test_path)


def partition_file_names(split_percentage, list_of_file_names):
    num_files = int(round(split_percentage * len(list_of_file_names)))
    shuffled = list_of_file_names[:]
    random.shuffle(shuffled)
    return shuffled[num_files:], shuffled[:num_files]


def get_3d_distance(pt1, pt2):
    return np.sqrt(np.sum((np.array(pt1) - np.array(pt2)) ** 2, axis=0))


def download_files(url, path):
    wget.download(url, path)


def git_clone_repo(url, path):
    Repo.clone(url, path=path)


def unzip_files(path):
    file_name = os.listdir(path)
    file = os.path.join(path, file_name[0])
    if file_name[0].endswith("tar.gz"):
        tar = tarfile.open(file, "r:gz")
        tar.extractall(path)
        tar.close()
    elif file_name[0].endswith(".zip"):
        with zipfile.ZipFile(file, 'r') as zip_ref:
            zip_ref.extractall(path)
    os.remove(file)


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)


def run_cmd(command):
    os.system(command)


def pip_install(package):
    pip.main(['install', package])


def create_label_map(labels, path):
    with open(path, 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')


def read_gip_images(path):
    gip_files = [pos_gip for pos_gip in os.listdir(path) if pos_gip.endswith('.gip')]
    color_images = []
    depth_images = []
    for i in range(0, len(gip_files), 2):
        depth_image = gip.read_image(os.path.join(path, gip_files[i])).pixel_data
        colour_image = gip.read_image(os.path.join(path, gip_files[i + 1])).pixel_data
        colour_image = np.array(cv2.cvtColor(colour_image, cv2.COLOR_BGR2RGB))
        color_images.append(colour_image)
        depth_images.append(depth_image)


def resize_predictions_detection_masks(predictions=None, image_shape=('height', 'width', 'channels'), mask_threshold=0.,
                                       verbose=1, debug=1):
    detection_masks = []

    for idx, detection_mask in enumerate(predictions['detection_masks']):
        height, width, _ = image_shape
        # background_mask with all black=0
        mask = np.zeros((height, width))

        # Get normalised bbox coordinates
        y_min, x_min, y_max, x_max = predictions['detection_boxes'][idx]

        # Convert to image size fixed bbox coordinates
        y_min, x_min, y_max, x_max = int(y_min * height), int(x_min * width), int(y_max * height), int(x_max * width)

        # Define bbox height and width
        bbox_height, bbox_width = y_max - y_min, x_max - x_min

        # Resize 'detection_mask' to bbox size
        bbox_mask = Image.fromarray(np.uint8(np.array(detection_mask) * 255), mode='L').resize(
            size=(bbox_width, bbox_height), resample=Image.NEAREST)

        # Insert detection_mask into image.size np.zeros((height, width)) background_mask
        mask[y_min:y_max, x_min:x_max] = bbox_mask
        mask_threshold = mask_threshold
        mask = np.where(mask > (mask_threshold * 255), 1, mask)
        # mask_threshold > 0.5 resulting mask seems too coarse, any value > 0 seems resulting in better masks
        # in case threshold is used to have other values (0)
        if mask_threshold > 0:
            mask = np.where(mask != 1, 0, mask)

        if debug:
            try:
                assert (
                        (np.unique(mask) == np.array([0., 1.])).all()  # in case, bbox_mask has (0,1)
                        or
                        (np.unique(mask) == np.array([0.])).all()  # in case, bbox_mask resulted in only (0)
                )
            except Exception as e:
                print(e)
                print('Mask Index:', idx)
                print('Mask unique values: ', np.unique(mask))
                print('Expected unique values: ', np.array([0., 1.]))
                break

        # Verbose first detection_mask
        if (verbose or debug) and idx == 0:
            print('Index (Example): ', idx)
            print('detection_mask shape():', np.array(detection_mask).shape)
            print('detection_boxes:', predictions['detection_boxes'][idx])
            print('detection_boxes@image.size (y_min, x_min, y_max, x_max) :', [y_min, x_min, y_max, x_max])
            print('detection_boxes (height, width):', (bbox_height, bbox_width))

        # Append mask to list() 'detection_masks'
        detection_masks.append(mask.astype(np.uint8))

    return detection_masks


def create_category_index(categories):
    """
    Creates dictionary of COCO compatible categories keyed by category id.

    Args:
      categories: a list of dicts, each of which has the following keys:
        'id': (required) an integer id uniquely identifying this category.
        'name': (required) string representing category name
          e.g., 'cat', 'dog', 'pizza'.

    Returns:
      category_index: a dict containing the same entries as categories, but keyed
        by the 'id' field of each category.
    """
    category_index = {}
    for cat in categories:
        category_index[cat['id']] = cat
    return category_index
