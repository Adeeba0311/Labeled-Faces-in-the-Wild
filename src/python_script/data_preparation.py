import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from python_script.feature_extraction import *
from sklearn.datasets import fetch_lfw_people
from sklearn.model_selection import train_test_split



def preparing_data(number_of_face):
    # Reading the dataset
    lfw_dataset = fetch_lfw_people(min_faces_per_person=number_of_face,color=True)


    # Preparing the histogram_oriented_gradients data
    image_rescaled = {}
    image_rescaled["original_image"] = np.array(lfw_dataset.images)
    flattend_arr = []
    hog_image = []
    for image in lfw_dataset.images:
        arr_2d = histogram_oriented_gradients(image)
        hog_image.append(arr_2d)
        flattend_arr.append(arr_2d.flatten())
        
    image_rescaled["hog_data"] = np.array(flattend_arr)
    image_rescaled["hog_image"] = np.array(hog_image)

    # Preparing the scale_invariant_feature_transform data
    flattend_arr = []
    sift_image = []
    for image in lfw_dataset.images:
        arr_2d = scale_invariant_feature_transform(image)
        sift_image.append(arr_2d)
        flattend_arr.append(arr_2d.flatten())

    image_rescaled["sift_data"] = np.array(flattend_arr)
    image_rescaled["sift_image"] = np.array(sift_image)


    # Preparing the local_binary_patterns data
    flattend_arr = []
    lbp_image = []
    for image in lfw_dataset.images:
        arr_2d = local_binary_patterns(image)
        lbp_image.append(arr_2d)
        flattend_arr.append(arr_2d.flatten())

    image_rescaled["lbp_data"] = np.array(flattend_arr)
    image_rescaled["lbp_image"] = np.array(lbp_image)

    return image_rescaled

