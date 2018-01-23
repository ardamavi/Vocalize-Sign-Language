# Arda Mavi
import sys
import numpy as np
from get_dataset import get_img
from scipy.misc import imresize
from database_process import get_data
from keras.models import model_from_json

image_size = 64
channel_size = 1

def predict(model, X):
    Y = model.predict(X)
    Y = np.argmax(Y, axis=1)
    Y = get_data('SELECT char FROM "id_char" WHERE id={0}'.format(Y[0]))
    return Y

if __name__ == '__main__':
    img_dir = sys.argv[1]
    img = 1-np.array(get_img(img_dir)).astype('float32')/255.
    img = img.reshape(1, image_size, image_size, channel_size)
    # Getting model:
    model_file = open('Data/Model/model.json', 'r')
    model = model_file.read()
    model_file.close()
    model = model_from_json(model)
    # Getting weights
    model.load_weights("Data/Model/weights.h5")
    print('Class:', predict(model, img)[0][0])
