# Vocalize Sign Language
### By Arda Mavi

Vocalize sign language with deep learning.

<img src="Assets/Alphabet Sign Language.jpg">

<img src="Assets/Numbers Sign Language.jpg">

In this project we use our own [Sign Language Digits Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset).

### Important Notes For Users:
- This project works best in the white background and good light.

### Additional:
<b>In this project, I added deep learning to my old lip reading project [SesimVar](https://github.com/ardamavi/SesimVar)(Turkish).</b>

## Running program:
Note: If you are failed, look up `For Development` title in bellow.

### Using Live Vocalize Command:
`python3 live.py`
Note: If you want, you can change the delay time.

### Using Predict Command:
`python3 predict.py <ImageFileName>`

# For Development:

## Getting Dataset:
Get [github.com/ardamavi/Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset) dataset and copy all files from `Sign-Language-Digits-Dataset/Dataset` folder to `Data/Train_Data`.

#### Artificial Intelligence Model Accuracy:
At the end of 10 epochs, 96% accuracy was achieved in the test without data augmentation.

### Model Architecture:
- Input Data
Shape: 64x64x1

- Convolutional Layer
32 filter
Filter shape: 3x3
Strides: 1x1
Padding: Same

- Activation
Function: ReLu

- Convolutional Layer
64 filter
Filter shape: 3x3
Strides: 1x1
Padding: Same

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2
Strides: 2x2

- Convolutional Layer
64 filter
Filter shape: 3x3
Strides: 1x1
Padding: Same

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2
Strides: 2x2

- Convolutional Layer
128 filter
Filter shape: 3x3
Strides: 1x1
Padding: Same

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2
Strides: 2x2

- Flatten

- Dense
Size: 526

- Activation
Function: ReLu

- Dropout
Rate: 0.4

- Dense
Size: 128

- Activation
Function: ReLu

- Dense
Size: Class size in dataset

- Activation
Function: Softmax

##### Optimizer: Adadelta
##### Loss: Categorical Crossentropy

### Model Training:
`python3 train.py`

### Using TensorBoard:
`tensorboard --logdir=Data/Checkpoints/logs`

### Creating DataBase:
- For getting ours dataset([Sign-Language-Digits-Dataset](https://github.com/ardamavi/Sign-Language-Digits-Dataset)) look up `Getting Dataset` title in this file.
- Create 'Data/Train_Data' folder.
- Create folder in 'Data/Train_Data' folder and rename what you want to add char or string.
- In your created char or string named folder add much photos about created char or string named folder.
Note: We work on 64x64 image also if you use bigger, program will automatically return to 64x64.

### Important Notes:
- Used Python Version: 3.6.0
- Install necessary modules with `sudo pip3 install -r requirements.txt` command.
- Install OpenCV (We use version: 3.2.0-dev)
