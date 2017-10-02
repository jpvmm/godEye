import glob
from PIL import Image
import numpy as np
from sklearn.cross_validation import train_test_split
from skimage.io import imread
from skimage.transform import resize
from skimage.feature import hog


def get_normal_data(class1, class2, class3, class4):
    ''' Open images augment and reshape it to the use of the CNN'''
    y = []
    images = []

    for image in sorted(glob.glob1(class1, "*.jpg")):
        print "Loading image from class1..." + image
        img = imread(class1+"/" + image, as_grey=True)
        img = resize(img,(200,200))

        images.append(img)
        y.append([0])

    for image2 in sorted(glob.glob1(class2, "*.jpg")):
        img2 = imread(class2+"/" + image2, as_grey=True)
        img2 = resize(img2,(200,200))
        print "Loading image from class2..." + image2
        images.append(img2)
        y.append([1])

    for image3 in sorted(glob.glob1(class3, "*.jpg")):
        img3 = imread(class3 + "/" + image3, as_grey=True)
        img3 = resize(img3, (200,200))
        print "Loading image from class3..." + image2

        images.append(img3)
        y.append([2])

    for image4 in sorted(glob.glob1(class4, "*.jpg")):
        img4 = imread(class4 + "/" + image4, as_grey=True)
        img4 = resize(img4,(200,200))
        print "Loading image from class4..." + image4

        images.append(img4)
        y.append([3])
    print "ALL FACES LOADED"

    return images, y

def reshape_data(images, y):
    images = np.array(images)
    images = images.reshape(images.shape[0],images.shape[1]*images.shape[2])

    y = np.array(y)
    y = y.ravel()

    X_train, X_test, y_train, y_test = train_test_split(images, y, test_size=0.25,
    random_state = 0)


    return X_train, X_test, y_train, y_test

def getHogFeatures(images):
    '''Extract hog features from the images'''
    hog_feat, hog_img = zip(*np.array([hog(img, orientations=9, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualise=True
                       ) for img in imgs]))

    return np.array(hog_feat), np.array(hog_img)

if __name__=="__main__":
    class1 = '/home/joao/Projects/godEye/picfaces/train/guilherme'
    class2 = '/home/joao/Projects/godEye/picfaces/train/joao'
    class3 = '/home/joao/Projects/godEye/picfaces/train/muria'
    class4 = '/home/joao/Projects/godEye/picfaces/train/others'

    imgs, y = get_normal_data(class1,class2,class3,class4)

    print "IMAGES LOADED"

    features, hog_images = getHogFeatures(imgs)

