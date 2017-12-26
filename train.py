# Arda Mavi
import os
import numpy
from get_dataset import get_dataset
from get_model import get_model, save_model
from keras.callbacks import ModelCheckpoint, TensorBoard

epochs = 10
batch_size = 5

def train_model(model, X, X_test, Y, Y_test):
    checkpoints = []
    if not os.path.exists('Data/Checkpoints/'):
        os.makedirs('Data/Checkpoints/')

    checkpoints.append(ModelCheckpoint('Data/Checkpoints/best_weights.h5', monitor='val_loss', verbose=0, save_best_only=True, save_weights_only=True, mode='auto', period=1))
    checkpoints.append(TensorBoard(log_dir='Data/Checkpoints/./logs', histogram_freq=1, write_graph=True, write_images=True, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None))

    '''
    # Creates live data:
    # For better yield. The duration of the training is extended.

    from keras.preprocessing.image import ImageDataGenerator
    generated_data = ImageDataGenerator(rotation_range=0.1,  width_shift_range=0.1, height_shift_range=0.1, horizontal_flip = True, vertical_flip = False)
    generated_data.fit(X)

    model.fit_generator(generated_data.flow(X, Y, batch_size=batch_size), steps_per_epoch=X.shape[0]/batch_size, epochs=epochs, validation_data=(X_test, Y_test), callbacks=checkpoints)

    '''
    model.fit(X, Y, batch_size=batch_size, epochs=epochs, validation_data=(X_test, Y_test), shuffle=True, callbacks=checkpoints)

    return model

def main():
    X, X_test, Y, Y_test = get_dataset()
    model = get_model(num_class=10)
    model = train_model(model, X, X_test, Y, Y_test)
    save_model(model)
    return model

if __name__ == '__main__':
    main()
