import csv
import cv2
import numpy as np
import tensorflow as tf

flags = tf.app.flags
FLAGS = flags.FLAGS

# command line flags
flags.DEFINE_string('data_dir', './default_data/', "Directory of training data")
flags.DEFINE_string('fine_tune', '', "File of model to fine-tune (.h5)")
flags.DEFINE_string('evaluate', '', "File of model to evaluate (.h5)")
flags.DEFINE_string('visualize', '', "File of model to visualize (.h5)")

# create visualization of model and quit (needs pydot & graphviz)
if FLAGS.visualize:
    from keras.models import load_model
    from keras.utils.visualize_util import plot

    model = load_model(FLAGS.visualize)
    plot(model, to_file='model.png', show_shapes=True)
    quit()

# open csv and save lines
lines = []
with open(FLAGS.data_dir+'driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)


# loop through csv data and save images/measurements
images = []
measurements = []
correction_factor=0.2
for line in lines[1:]:
    for i in range(3):
        source_path = line[i]
        filename = source_path.split('/')[-1]
        current_path = FLAGS.data_dir+'IMG/' + filename
        image = cv2.cvtColor(cv2.imread(current_path), cv2.COLOR_BGR2RGB)
        if i == 0:
            measurement = float(line[3])
        elif i==1:
            measurement = float(line[3])+correction_factor
        elif i==2:
            measurement = float(line[3])-correction_factor
        # less straight-driving data
        if measurement==0 and np.random.rand()<0.2:
            continue
        images.append(image)
        measurements.append(measurement)

# augment the data (flip images)
augmented_images, augmented_measurements = [], []
for image, measurement in zip(images, measurements):
    augmented_images.append(image)
    augmented_measurements.append(measurement)
    augmented_images.append(cv2.flip(image,1))
    augmented_measurements.append(-measurement)

# create arrays for training
X_train = np.array(augmented_images)
y_train = np.array(augmented_measurements)

# load model and fine-tune if flagged
if FLAGS.fine_tune:
    from keras.models import load_model
    from keras.optimizers import SGD

    model = load_model(FLAGS.fine_tune)

    model.compile(loss='mse', optimizer=SGD(lr=0.0001))
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=1)

    model.save('model.h5')

elif FLAGS.evaluate: # evaluate model without training
    from keras.models import load_model

    model = load_model(FLAGS.evaluate)

    loss = model.evaluate(X_train, y_train) # uses entire dataset
    print(loss)

else: # create a new model from scratch & train
    from keras.models import Sequential
    from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
    from keras.layers.convolutional import Convolution2D
    from keras.layers.pooling import MaxPooling2D

    model = Sequential()
    model.add(Cropping2D(cropping=((70,25), (0,0)), input_shape=(160,320,3)))
    model.add(Lambda(lambda x: x / 255.0 - 0.5))
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Convolution2D(6,5,5,activation='relu'))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(84, activation='relu'))
    model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X_train, y_train, validation_split=0.2, shuffle=True, nb_epoch=5)

    model.save('model.h5')

# make an annoying sound
for i in range(10):
    print('\a')
