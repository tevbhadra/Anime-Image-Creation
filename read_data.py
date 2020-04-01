#reads the data from disk
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob


def readImagesShort(path, scale_size = 0, count = None):
    imageList = []

    for imPath in os.listdir(path):
        if os.path.isdir(os.path.join(path, imPath)):
            imagesPath = os.path.join(path, imPath)
        else:
            continue
        print("Reading images from ", path+imPath)
        for image in glob(imagesPath+"\\*.png"):
            im = cv2.imread(image)
            if not (scale_size == 0):
                im = cv2.resize(im, (scale_size, scale_size))
            imageList.append(im)

            if count != None:
                count = count - 1
                if count == 0:
                    return np.array(imageList)


    return np.array(imageList)
