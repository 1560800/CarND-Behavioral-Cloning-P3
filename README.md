# Behavioral Cloning Project

In this project, I use a neural network to clone car driving behavior. It is a supervised regression problem between the car steering angles and the road images in front of a car.  
## Project Description
### The goals / steps of this project are the following:
- Build, a convolution neural network in Keras that predicts steering angles from images
- Train and validate the model with a training and validation set
- Test that the model successfully drives around track one without leaving the road
- Summarize the results with a written report  
### Files included
`model.py` : The script used to create and train the model  
`drive.py` : The script to drive the car in autonomous mode  
`model.h5` : The script to provide useful functionalities (i.e. image preprocessing and augumentation)  
`writeup_report.md` : Summarize the results   
### Getting started
Additionally you need to download and unpack the [Udacity self-driving car simulator (Version 2)](https://github.com/udacity/self-driving-car-sim).    
### Run the pretrained model
To run the code start the simulator in `autonomous mode`, Then, run the model as follows:  
```sh
python drive.py model.h5
```

### To train the model
first make a directory ./data/, drive the car in training mode around the track and save the data to this directory.   
Then, run the model as follows:
```sh
python model.py
```
## Model Architecture and Training Strategy
### Model architecture
I used the NVIDIA's CNN model introduced in the Udacity lesson.  
 
 
|Layer (type)          |       Output Shape         |     Param    |  
|----------------------|:--------------------------:|:-------------:|  
|lambda_1 (Lambda)     |       (None, 45, 160, 3)   |     0        |   
|conv2d_1 (Conv2D)     |       (None, 21, 78, 64)   |     4864     |  
|conv2d_2 (Conv2D)     |       (None, 9, 37, 36)    |     57636    | 
|conv2d_3 (Conv2D)     |       (None, 3, 17, 48)    |     43248    | 
|conv2d_4 (Conv2D)     |       (None, 1, 15, 64)    |     27712    | 
|dropout_1 (Dropout)   |       (None, 1, 15, 64)    |     0        | 
|flatten_1 (Flatten)   |       (None, 960)          |     0        | 
|dense_1 (Dense)       |       (None, 100)          |     96100    | 
|dense_2 (Dense)       |       (None, 50)           |     5050     | 
|dense_3 (Dense)       |       (None, 10)           |     510      | 
|dense_4 (Dense)       |       (None, 1)            |     11       | 

Total params: 235,131
Trainable params: 235,131

### Data Preprocessing  
First, read images and steering angles, which are the dataset provided from Udacity, from the csv file.
Then converted the color space from BGR into RGB for drive.py, cropped top 50[pixel] and bottom 20[pixel] (i.e. 160x320x3 ---> 90x320x3), and resized to 45x160 (i.e. 90x320x3 ---> 45x160x3) in order to feed to the NVIDIA's CNN architecture.



