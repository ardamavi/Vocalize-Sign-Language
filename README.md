# Vocalize-Sign-Language
### By Arda Mavi

Vocalize sign language with deep learning.

### Using Live Vocalize Command:
`python3 live.py`

### Using Predict Command:
`python3 predict.py <ImageFileName>`

### Model Training:
`python3 train.py`

### Using TensorBoard:
`tensorboard --logdir=Data/Checkpoints/logs`

### Model Architecture:
- Input Data
Shape: 150x150x3

- Convolutional Layer
32 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: ReLu

- Convolutional Layer
64 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: ReLu

- Convolutional Layer
64 filter
Filter shape: 3x3
Strides: 1x1

- Activation
Function: ReLu

- Max Pooling
Pool shape: 2x2
Strides: 2x2

- Flatten

- Dense
Size: 1280

- Activation
Function: ReLu

- Dropout
Rate: 0.5

- Dense
Size: Class size in database

- Activation
Function: Sigmoid

##### Optimizer: Adadelta
##### Loss: Categorical Crossentropy

#### Used Python Version: 3.6.0

### Creating DataBase:
- Create 'Data/Train_Data' folder.
- Create folder in 'Data/Train_Data' folder and rename what you want to add char or string.
- In your created char or string named folder add much photos about created char or string named folder.
Note: We work on 150x150 image also if you use bigger, program will automatically return to 150x150.

<b>In this project, I added deep learning to my old lip reading project [SesimVar](https://github.com/ardamavi/SesimVar)(Turkish).</b>
