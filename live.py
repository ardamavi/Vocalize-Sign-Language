# Arda Mavi
import os
import cv2
import platform
import numpy as np
from predict import predict
from scipy.misc import imresize
from multiprocessing import Process
from keras.models import model_from_json

img_size = 64
grayscale_images = True # False: RGB

def main():
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")

    # Get image from camera, get predict and say it with another process, repeat.
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        cv2.imshow('Arda Mavi',img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
        img = np.array(img).astype('float32')/255.
        img = img.reshape(1, img_size, img_size, 1 if grayscale_images else 3)
        Y = predict(model, img)[0][0]
        if(platform.system() == 'Darwin'):
            arg = 'say {0}'.format(Y)
            # Say predict with multiprocessing
            Process(target=os.system, args=(arg,)).start()
        if cv2.waitKey(2000) == 27: # Decimal 27 = Esc
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
