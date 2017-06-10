# Arda Mavi
import cv2
import platform
from predict import predict
from multiprocessing import Process
# TODO: Get image from camera, get predict and say it with another process, repeat.

def main():
    cap = cv2.VideoCapture(0)
    while 1:
        ret, img = cap.read()
        cv2.imshow('Arda Mavi',img)
        Y = predict(img)
        print('Predict: {0}'.format(Y))
        if(platform.system() == 'Darwin'):
            arg = 'say {0}'.format(Y)
            # Say predict with multiprocessing
            Process(target=os.system, args=(arg,)).start()
        if cv2.waitKey(1) == 27: # Decimal 27 = Esc
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
