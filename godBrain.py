import cv2 

camera = cv2.VideoCapture(0)

i = 0 

while i < 1:
    raw_input('Pressione qualquer tecla para capturar')
    return_value, imagem = camera.read()
    cv2.imwrite('opencv'+str(i)+'.jpg', imagem)
    i = i+1

del(camera)
