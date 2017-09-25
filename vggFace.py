import numpy as np
from skimage import color
from PIL import Image
import glob
import matplotlib.pyplot as plt
from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Reshape, Permute, Activation, \
    Input, merge, GlobalAveragePooling2D
from keras.layers.convolutional import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

train_dir = "/home/joao/Academia/projects/godEye/DATA/picfaces/train"
val_dir = "/home/joao/Academia/projects/godEye/DATA/picfaces/val"

def vgg_face(weights_path=None):
    #datagen = ImageDataGenerator(rescale = 1./255, samplewise_center=True)
    
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


def save_bottleneck_features(model,train_data_dir,val_data_dir,batch_size):
    '''Save the features extracted by the model'''

    datagen = ImageDataGenerator(rescale = 1./255)

    generator = datagen.flow_from_directory(train_data_dir, target_size=(224,224),batch_size=batch_size,
            class_mode = None, shuffle=False)
    bottleneck_features_train = model.predict_generator(generator,24)
    np.save(open('bottleneck_features_train.npy','w'), bottleneck_features_train)
    trainLabels = generator.classes
    np.save(open('train_labels.npy','w'), trainLabels)
    print 'Train features saved!'

    

    val_generator = datagen.flow_from_directory(
        val_data_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)
    bottleneck_features_validation = model.predict_generator(val_generator, 19)
    np.save(open('bottleneck_features_validation.npy', 'w'), bottleneck_features_validation)
    valLabels = val_generator.classes
    np.save(open('validation_labels.npy','w'), valLabels)
    print 'Val features saved!'

    print generator.class_indices

def open_features():
    '''Open the features saved by the CNN model'''
    train_data = np.load(open('bottleneck_features_train.npy'))
    train_labels = np.load(open('train_labels.npy'))

    val_data = np.load(open('bottleneck_features_validation.npy'))
    val_labels = np.load(open('validation_labels.npy'))

    return train_data, train_labels, val_data, val_labels


def train_top_model(train_data, train_labels, val_data, val_labels):
    '''Train the FC layers'''
    batch_size = 1

    NN = Sequential()
    NN.add(Flatten(input_shape=train_data.shape[1:]))
    NN.add(Dense(512, activation='relu'))
    NN.add(Dropout(0.5))
    NN.add(Dense(256, activation='relu'))
    NN.add(Dropout(0.5))
    NN.add(Dense(1, activation='sigmoid'))

    NN.compile(optimizer='rmsprop',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    NN.fit(train_data, train_labels,
          epochs=50, batch_size=batch_size,
          validation_data=(val_data,val_labels))
    return NN

def train_top_categorical(train_data, train_labels, val_data, val_labels):
    '''Train the FC layers'''
    batch_size = 1

    NNC = Sequential()
    NNC.add(Flatten(input_shape=train_data.shape[1:]))
    NNC.add(Dense(512, activation='relu'))
    NNC.add(Dropout(0.5))
    NNC.add(Dense(256, activation='relu'))
    NNC.add(Dropout(0.5))
    NNC.add(Dense(3, activation='softmax'))

    NNC.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    NNC.fit(train_data, train_labels,
          epochs=50, batch_size=batch_size,
          validation_data=(val_data,val_labels))
    return NNC

def buildModel(extractor):
    "Build the model based on vggFace extractor"
    
    for layer in extractor.layers:
        layer.trainable = False

    x = extractor.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(523, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    predictions = Dense(3, activation='softmax')(x) #new softmax layer
    model = Model(input=extractor.input, output=predictions)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def get_nb_files(directory):
  """Get number of files by searching directory recursively"""
  if not os.path.exists(directory):
    return 0
  cnt = 0
  for r, dirs, files in os.walk(directory):
    for dr in dirs:
      cnt += len(glob.glob(os.path.join(r, dr + "/*")))
    return cnt


def trainTransfer(model,train_dir, val_dir):
    """ Train the model built for transfer learning """
    train_samples = get_nb_files(train_dir)
    val_samples = get_nb_files(val_dir)

    train_datagen =  ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )
    test_datagen = ImageDataGenerator(
            rotation_range=30,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True
            )   
    train_generator = train_datagen.flow_from_directory(train_dir,
            target_size = (224,224),
            batch_size = 2)
    test_generator = test_datagen.flow_from_directory(val_dir,
            target_size = (224,224),
            batch_size = 2)

    model.fit_generator(train_generator,
            nb_epoch = 50,
            samples_per_epoch = train_samples,
            validation_data = test_generator,
            nb_val_samples = val_samples,
            class_weight='auto')



