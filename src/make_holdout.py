# Modified from Michael Dyer's script

from numpy.random import choice
import cv2
import os
import numpy as np


def train_val_holdout_split_images(root_path, data_dir, train_ratio  = 0.85, validation_ratio = 0.0, holdout_ratio = 0.15):
    """
    Utility to split categorical training files organized by folder into training and testing, with resizing and max_images
    Args:
        root_path (str): folder containing category folders
        n_images (int): max number of images to copy
        train_ratio (float): ratio of images to copy to train folder
        validation_ratio(float): ratio of images to copy to validation folder
        holdout_ratio (float): ratio of images to copy to holdout folder
        resize_size (tuple(int, int)): size in pixels to resize copied images
    Returns:
        None
        """
    root_path_lite = root_path

    train_folder      = root_path + data_dir + '/train'
    validation_folder = root_path + data_dir + '/validation'
    holdout_folder    = root_path + data_dir + '/holdout'

    root_path = root_path + '/raw_3c'

    resolutions = np.ones((1,2))

    for root, dirs, files in os.walk(root_path, topdown=False):
        for name in files:
            ext = name.split('.')[-1]
            if ext in ['jpg','jpeg','png']:
               current_path = os.path.join(root, name)
               root_dir, category = os.path.split(root)
               val_split_dir = choice ([train_folder, validation_folder, holdout_folder], 1, p =[train_ratio, validation_ratio, holdout_ratio])[0]
               new_dir = os.path.join(val_split_dir, category)
               if not os.path.exists(new_dir):
                   os.makedirs(new_dir)
               new_path = os.path.join(new_dir, name)
               o_img = cv2.imread(current_path)
               cv2.imwrite(new_path, o_img)
               try:
                   resolutions = np.vstack((resolutions,o_img.shape[:2]))
               except:
                   print(f'Resolution failed on {current_path}!!!!!!')
                   input = Input('Press Enter to Continue')
               print(new_path)
    print(f'Mean Img Size: {resolutions[1:].mean(axis=0)}')
    print(f'Stdev Img Size: {resolutions[1:].std(axis=0)}')
    print(f'Min Img Size: {resolutions[1:].min(axis=0)}')
    print(f'Max Img Size: {resolutions[1:].max(axis=0)}')


train_val_holdout_split_images('../data/Banana_People_Not','/3_Classes')
