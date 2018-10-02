import cv2 
import numpy as np

from open_data import *
import cPickle

import matplotlib.pyplot as plt

from abrePorta import *

def captureFace():
    c = cv2.VideoCapture(0)
    i = 0
    print "Pressione ESPACO para iniciar a identificacao"
    print "Pressione ESC para sair do programa"

    
    while(1):
        i = i +1
        _,frame = c.read()

        cv2.imshow('e1', frame)


        k = cv2.waitKey(33)

        if k == 27:
            print "Usuario quer sair do programa"
            break
        if k == 32:
            print "Imagem capturada!"
            cv2.imwrite('opencv'+str(i)+'.jpg',frame)
            break

    c.release()
    cv2.destroyAllWindows()
    
    new_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    return new_frame

def extractFace(new_frame):
    casc_file_src = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(casc_file_src)

    faces = faceCascade.detectMultiScale(new_frame,1.2,5)
    sub_face = None

    for (x,y,w,h) in faces:
        cv2.rectangle(new_frame, (x,y), (x+w, y+h),(0,255,0),2)
        sub_face = new_frame[y:y+h, x:x+w]
        sub_face = cv2.resize(sub_face,(200,200))

    return sub_face

def load_svm():
    with open('oneVsRest_svm.pkl', 'rb') as f:
        svm = cPickle.load(f)

    return svm

def predict_user(svm, features):
    features = features.reshape(1,features.shape[0])
    
    user = svm.predict(features)

    if user == 0:
        print 'Bem vindo Guilherme!'
        print 'Entrada liberada!'
        abre_porta()
    elif user == 1:
        print 'Bem vindo Joao!'
        print 'Entrada liberada!'
        abre_porta()
    elif user == 2:
        print 'Bem vindo Muria!'
        print 'Entrada liberada!'
        abre_porta()
    else:
        print 'Pessoa nao identificada, ACESSO NEGADO!'




if __name__=='__main__':
    face = captureFace()
    sub_face = extractFace(face)
    if sub_face == None:
        print 'Rosto NAO  identificado!'
    else:
        print 'Rosto identificado!'
        features, hog_images = getSingleHogFeatures(sub_face)
        plt.imshow(sub_face)
        plt.show()
        plt.imshow(hog_images)
        plt.show()
        svm = load_svm()
        predict_user(svm, features)

