
# coding: utf-8

# In[1]:

import numpy as np
from skimage import color
from PIL import Image
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation,     Input, merge
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img


# In[4]:

def vgg_face(weights_path=None):
    datagen = ImageDataGenerator(rescale = 1./255, samplewise_center=True)
    
    #Build the VGG16 Net
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))

    model.add(Conv2D(64,(3,3), activation='relu', name='conv1_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(64,(3,3), activation='relu', name='conv1_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3,3), activation='relu', name='conv2_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(128,(3,3), activation='relu', name='conv2_2'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3,3), activation='relu', name='conv3_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3,3), activation='relu', name='conv3_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(256,(3,3), activation='relu', name='conv3_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3), activation='relu', name='conv4_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3), activation='relu', name='conv4_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3), activation='relu', name='conv4_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3), activation='relu', name='conv5_1'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3), activation='relu', name='conv5_2'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Conv2D(512,(3,3), activation='relu', name='conv5_3'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
   
    
    if weights_path:
        model.load_weights(weights_path, by_name=True)
        print "Model loaded."
        
    return model




# In[5]:

weightsPath="/home/joao/Projects/godEye/vgg-face-keras-fc.h5"
model = vgg_face(weightsPath)


# In[6]:

im = Image.open('A.J._Buckley.jpg')
im = im.resize((224,224))
im = np.array(im).astype(np.float32)
#    im[:,:,0] -= 129.1863
#    im[:,:,1] -= 104.7624
#    im[:,:,2] -= 93.5940
im = im.transpose((2,0,1))
im = np.expand_dims(im, axis=0)




# In[12]:

def load_fc(weights_path=None):
    model.add(Dense(4096, activation='relu', name='fc6'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu',name='fc7'))
    model.add(Dropout(0.5))
    model.add(Dense(2622, activation='softmax',name='fc8'))
    
    if weights_path:
        model.load_weights(weights_path, by_name=True)
        print "Model loaded."
    return model


# In[17]:

fc = load_fc(weightsPath)


# In[16]:

fc


# In[ ]:



