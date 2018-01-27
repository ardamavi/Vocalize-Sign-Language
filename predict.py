# Arda Mavi
import sys
import numpy as np
from get_dataset import get_img
from scipy.misc import imresize
from database_process import get_data
from keras.models import model_from_json

image_size = 64
channel_size = 1

def predict(model, X): # Return: Y String , Y Possibility
    Y = model.predict(X)
    Y_index = np.argmax(Y, axis=1)
    Y_string = get_data('SELECT char FROM "id_char" WHERE id={0}'.format(Y_index[0]))
    return Y_string[0][0], Y[0][Y_index][0]

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
    Y_string, Y_possibility = predict(model, img)
    print('Class:', Y_string, '\nPossibility:', Y_possibility)
