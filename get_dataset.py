# Arda Mavi
import os
import numpy as np
from os import listdir
from scipy.misc import imread, imresize
from keras.utils import to_categorical
from database_process import create_table, add_data
from sklearn.model_selection import train_test_split

# Settings:
img_size = 64
grayscale_images = True # False: RGB
num_class = 10
test_size = 0.2


def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path, flatten=grayscale_images)
    img = imresize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img

def get_dataset(dataset_path='Data/Train_Data'):
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_dataset/X.npy')
        Y = np.load('Data/npy_dataset/Y.npy')
    except:
        if os.path.exists('Data/DataBase/database.sqlite'):
            os.remove('Data/DataBase/database.sqlite')
        # Create database:
        create_table('id_char','id, char')
        labels = listdir(dataset_path) # Geting labels
        X = []
        Y = []
        count_categori = [-1,''] # For encode labels
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                # For encode labels:
                if label != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = label
                    add_data('id_char', "{0}, '{1}'".format(count_categori[0], count_categori[1]))
                Y.append(count_categori[0])
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        X = X.reshape(X.shape[0], img_size, img_size, 1 if grayscale_images else 3)
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('Data/npy_dataset/'):
            os.makedirs('Data/npy_dataset/')
        np.save('Data/npy_dataset/X.npy', X)
        np.save('Data/npy_dataset/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=test_size, random_state=42)
    return X, X_test, Y, Y_test

if __name__ == '__main__':
    get_dataset()
