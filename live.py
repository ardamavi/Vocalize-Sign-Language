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
channel_size = 1

def main():
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    
    print('Press "ESC" button for exit.')

    # Get image from camera, get predict and say it with another process, repeat.
    cap = cv2.VideoCapture(0)
    old_char = ''
    while 1:
        ret, img = cap.read()
        
        # Cropping image:
        img_height, img_width = img.shape[:2]
        side_width = int((img_width-img_height)/2)
        img = img[0:img_height, side_width:side_width+img_height]
        
        # Show window:
        cv2.imshow('VSL', cv2.flip(img,1)) # cv2.flip(img,1) : Flip(mirror effect) for easy handling.
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = imresize(img, (img_size, img_size, channel_size))
        img = 1-np.array(img).astype('float32')/255.
        img = img.reshape(1, img_size, img_size, channel_size)
        
        Y_string, Y_possibility = predict(model, img)
        
        if Y_possibility < 0.4: # For secondary vocalization
            old_char = ''
        
        if(platform.system() == 'Darwin') and old_char != Y_string and Y_possibility > 0.6:
            print(Y_string, Y_possibility)
            arg = 'say {0}'.format(Y_string)
            # Say predict with multiprocessing
            Process(target=os.system, args=(arg,)).start()
            old_char = Y_string
        if cv2.waitKey(200) == 27: # Decimal 27 = Esc
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
