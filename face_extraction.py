import cv2
import matplotlib.pyplot as plt
import numpy as np
import glob

#TODO: verificar se eh melhor passar so o array da imagem o path da pasta para extracao dos rostos

def extractFacesFromFolder(folderPath):
    '''Will extract the face from the photo'''

    cascade_file_src = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascade_file_src)
    

    for filename in glob.glob(folderPath+'*.jpg'):
            
        # load image on gray scale :
        image = cv2.imread(filename)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # detect faces in the image :
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)

        # draw rectangles around the faces :
        for (x, y, w, h) in faces:
            cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            sub_face = image[y:y+h, x:x+w]
        
        cv2.imwrite(filename+'face'+'.jpg',sub_face)
    #return sub_face

def extractFaceFromFile(filePath):
    '''Will extract the face from a single photo'''

    cascade_file_src = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(cascade_file_src)
    image = cv2.imread(filePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(gray, 1.2, 5)

    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        sub_face = image[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face,(224,224))
    
    return sub_face
    
