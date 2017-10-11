# **Behavioral Cloning**
Project by Patrick Klein for Udacity's Self-driving Car Nanodegree

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./model.png "Model Visualization"
[image2]: ./examples/left_2016_12_01_13_35_49_125.jpg "Example, left"
[image3]: ./examples/center_2016_12_01_13_35_49_125.jpg "Example, center"
[image4]: ./examples/right_2016_12_01_13_35_49_125.jpg "Example, right"
[image5]: ./examples/center_cropped.jpg "Example, cropped"
[image6]: ./examples/center_flipped.jpg "Example, flipped"
[image7]: ./examples/placeholder_small.png "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following required files:
* **model.py** containing the script to create and train the model
* **drive.py** for driving the car in autonomous mode
* **model.h5** containing the final trained convolution neural network
* **video.mp4** with a recording of two laps in autonomous mode using the final model
* **writeup_report.md** summarizing these results

#### 2. Submission includes functional code
Using the Udacity provided simulator and the drive.py file, the car can be driven autonomously around the track by executing

```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

Even though I trained on a dataset with nearly 40k images, I did not implement a Python generator.  My computer has 16GB RAM and training only required a max of ~13.6GB of memory.  Total train time takes less than an hour.

---

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

The model used in this project is a convolutional neural network, built using Keras, with a design similar to LeNet-5.  A more detailed visualization of the network is presented further in the report.

#### 2. Attempts to reduce overfitting in the model

In order to prevent overfitting and generalize the model, the dataset of images were augmented to include flipped images.  Additionally, the datasets were split between training and testing, with 20% reserved for the validation set.  Furthermore, even though the model underwent fine-tuning after initial training, the model was never trained on the same dataset more than once to ensure validation images didn't contaminate the training sets.

Overfitting was not a significant concern for this project because I used a relatively small model for the given task (as compared to [NVIDIA's network](http://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf)).  And because it is not strictly necessary for the model to generalize (since only one track was being tested), having the model "memorize" the track by overfitting would not actually prevent a successful project in this case.

#### 3. Model parameter tuning

The model used an Adam optimizer during the initial training, which uses adaptive learning rates for each parameter.  Initial training lasted for 5 fixed epochs.

When refining the model on new training images, an SGD optimizer was used with a small learning rate of 0.0001.  A small learning rate was used with only 1 epoch in order to prevent overfitting on the new images.

#### 4. Appropriate training data

The initial training used the sample training data provided by Udacity.  The model created from this data was saved as `model-1.h5`.

The images for a second training phase were collected by me.  This dataset included normal driving on track one, with one lap clockwise and one lap counter-clockwise.  I used a keyboard for steering.  The model trained on this data was saved as `model-2.h5`, and is also used for the final submission as just `model.h5`.

A third set of images were also collected, which included two laps on track two, using a joystick for smoother steering measurements.  However, this did not increase the quality of the model's driving on track one.  The model trained on this set can be found on my repo as `model-3.h5`.

All of the training sets were augmented with flipped images, and corrected for BGR to RGB.  Left and right images were also used, using a correction factor of +/- 0.2.  In order to avoid a bias for straight driving, I include a 20% probability of skipping images that were recored with a measurement of 0.

---

### Architecture and Training Documentation

#### 1. Solution Design Approach

When initially designing the model, I closely followed the suggestions from the Udacity lessons.  The model has a design similar to LeNet-5.  The main difference is that I included cropping and normalization into the model, and there is only one linear output.

I was surprised at how well the model performed after only one training epoch on the sample training data.  It mostly followed the road up until the bridge, at which point it drove into the sides.  I then increased training to five epochs, and saw that the model mostly stayed on the road, with the only real exception being the curve after the bridge.

At this point I settled on the model, and focused on quality of the training data, as outlined below.  There are many ways the model could be improved (dropout/more layers/more kernels/etc.) but this model was sufficient for this project.

#### 2. Final Model Architecture

Here is a detailed description of the model.  A Graphviz visualization can also be seen [here](model.png).

| Layer                        |     Description                             | Output Size |
|:----------------------------:|:-------------------------------------------:|:--------:|
| Input                        | RGB image   							                   |160x320x3 |
| Cropping                     | 70 pixels from top , 25 pixels from bottom  |65x320x3  |
| Normalization                | [-0.5, 0.5]                                 |65x320x3  |
| 2D Convolution + ReLU        | 6 5x5 kernels                               |61x316x6  |
| Max Pooling                  | 2x2 pool                                    |30x158x6  |
| 2D Convolution + ReLU        | 6 5x5 kernels                               |26x154x6  |
| Max Pooling                  | 2x2 pool                                    |13x77x6   |
| Flatten                      |                                             |6006      |
| Fully Connected + ReLU       |                                             |120       |
| Fully Connected + ReLU       |                                             |84        |
| Fully Connected              |                                             |1         |

#### 3. Creation of the Training Set & Training Process

Initial training used the Udacity sample images.  This dataset includes 8,036 images on track one.  The driving in these images appears to maintain center-lane driving, and include clockwise and counter-clockwise laps. Here is an example image from this set:

![alt text][image3]

The dataset also includes images shifted to the left and right.  These were used during training as well, with a factor of 0.2 used to correct the steering angle.

![alt text][image2]
![alt text][image4]

To get the most out of this dataset, I wanted to augment the set of images.  The most practical was to do this was to flip copies of the images using `cv2.flip(image,1)` and take the negative of the steering angle.  Here is an example of one of the flipped images:

![alt text][image6]

For the pipeline, I decided to crop the images to help speed up training, as well as reduce the amount of irrelevant information in the image.  The model crops 70 pixels from the top and 25 pixels from the bottom.  Here is an example below:

![alt text][image5]

To avoid a bias in straight driving, I added a 20% chance that images with a steering angle of 0 be skipped when loading.  This seems to cause the model to veer back-and-forth on straight sections of road, but it helps the model turn harder when it is approaching the edge of the road.

Another important change in loading the images was to correct the color channels.  For example, OpenCV2 loads images as BGR, whereas the simulator reads in RGB.  I was able to convert the training images to RGB after loading them with OpenCV2 by using the command `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)`.  This greatly improved the ability of the model to recognize the boundaries of the road in the simulator, and was not an easy bug to spot since the validation loss would have remained low even with the bug.

After initial training, the model only drove onto the curb once when running on the simulator.  Because it was performing relatively well and I wanted to get experience with fine-tuning, I decided to just collect more data and update the parameters of my trained model.  I recorded an additional two laps on track one, both clockwise and counter-clockwise.  I fit my existing model to these images using the same pipeline.  The resulting model was able to pass the requirements of the project at this point.
