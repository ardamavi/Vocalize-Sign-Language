# Arda Mavi
import os
import numpy as np
from os import listdir
from skimage import io
from scipy.misc import imresize
from keras.utils import to_categorical
from database_process import create_table, add_data
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import array_to_img, img_to_array, load_img

def get_img(data_path):
    # Getting image array from path:
    img = io.imread(data_path)
    img = imresize(img, (150, 150, 3))
    return img

def get_dataset(dataset_path='Data/Train_Data'):
    # Create database:
    create_table('id_char','id, char')
    # Getting all data from data path:
    try:
        X = np.load('Data/npy_train_data/X.npy')
        Y = np.load('Data/npy_train_data/Y.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        len_datas = 0
        for label in labels:
            len_datas += len(listdir(dataset_path+'/'+label))
        X = np.zeros((len_datas, 150, 150, 3), dtype='float64')
        Y = np.zeros(len_datas, dtype=np.str)
        count_data = 0
        count_categori = [-1,''] # For encode labels
        for label in labels:
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X[count_data] = img
                # For encode labels:
                if label != count_categori[1]:
                    count_categori[0] += 1
                    count_categori[1] = label
                    add_data('id_char', "{0}, '{1}'".format(count_categori[0], count_categori[1]))
                Y[count_data] = count_categori[0]
                count_data += 1
        # Create dateset:
        Y = to_categorical(Y, count_categori[0]+1)
        if not os.path.exists('Data/npy_train_data/'):
            os.makedirs('Data/npy_train_data/')
        np.save('Data/npy_train_data/X.npy', X)
        np.save('Data/npy_train_data/Y.npy', Y)
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.1, random_state=42)
    return X, X_test, Y, Y_test
