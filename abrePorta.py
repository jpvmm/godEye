import time
import RPi.GPIO as GPIO



def abre_porta():
    GPIO.setmode(GPIO.BOARD)

    GPIO.setup(40,GPIO.OUT)

    GPIO.output(40, True)

    p = GPIO.PWM(40,50)
    p.start(7.5)
    
    p.ChangeDutyCycle(7.5)
    time.sleep(10)
    p.ChangeDutyCycle(2.5)
    time.sleep(5)
