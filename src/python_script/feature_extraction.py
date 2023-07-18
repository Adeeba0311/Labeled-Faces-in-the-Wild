import cv2
import numpy as np
from skimage import data
from skimage.feature import hog
from skimage import data, exposure
from skimage.color import label2rgb
from skimage.transform import rotate
from skimage.feature import local_binary_pattern


def histogram_oriented_gradients(image):
    # Example usage on a single image
    fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16),
                        cells_per_block=(1, 1), visualize=True, multichannel=True)

    # Visualize the HOG image
    hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
    return hog_image_rescaled
  
def scale_invariant_feature_transform(image):
    
    # Convert the image depth to CV_8U
    image8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
    
    #reading image
    gray1 = cv2.cvtColor(image8bit, cv2.COLOR_BGR2GRAY)
    
    #keypoints
    #keypoints``
    sift = cv2.SIFT_create()
    keypoints_1, descriptors_1 = sift.detectAndCompute(gray1, None)

    img_1 = cv2.drawKeypoints(gray1,keypoints_1,image)
    return img_1
    
def local_binary_patterns(image):
    
    # settings for LBP
    radius = 3
    n_points = 8 * radius
    
    # Convert the image depth to CV_8U
    image8bit = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    # Convert the array to grayscale and reshape it to (height, width)
    gray1 = cv2.cvtColor(image8bit, cv2.COLOR_BGR2GRAY)

    lbp = local_binary_pattern(gray1, n_points, radius, method = "uniform")

    return lbp