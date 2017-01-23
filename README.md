## Project: Traffic Sign Classification -- A 98.1% Solution Using Keras
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

### Overview

This project implements a solution to classify (German) Road Traffic Signs using Deep Learning
techniques, a project for Udacity's Self-Driving Car Program.

The current state of the art solutions for this problem are very accurate:

| Model        |  Accuracy     |  Authors |
| -------------|:-------------:|---------:|
| Committee of CNNs | 99.47% | [Sermanet et al] (http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf)|
| Multi-column DeepNNs | 99.45% | [Ciresan et al] (http://people.idsia.ch/~juergen/nn2012traffic.pdf)
| Human Accuracy | 98.32%      | [Humans](http://benchmark.ini.rub.de/)   |

The dataset can be obtained [here]((https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip)

I implemented this project using [Keras](http://keras.io) (running on top of TensorFlow) instead of on bare TensorFlow because (a) Keras provides a higher-level abstraction than barebones TensorFlow, (b) I've used TensorFlow previously and was keen to explore Keras's capabilities.

I've greatly benefited from a few people, including [Vivek Yadav][https://medium.com/@vivek.yadav/improved-performance-of-deep-learning-neural-network-models-on-traffic-sign-classification-using-6355346da2dc#.9scb6m7cf], whose solution inspired me for this write-up. _Thanks!_


1. Exploring the dataset
2. Data preprocessing / augmentation
3. Model architecture
4. Training
5. Model performance

---

### 1. Exploratory data analysis

The dataset contains about 37,000 German road traffic signs categorized among 43 different classes.
These classes are what you would normally expect, for example, Speed Limit 50 Kmph, Stop sign, Yield,
etc. They also somewhat less known signs such as Wild Animals Crossing, etc. The images are 32x32 pixel
each. However, the images are not evenly distributed. 

The following picture shows the distribution of the Training dataset and Test dataset. 
![png](images/train-test-distro.jpg)

As can be seen, some classes have ~2000 images while others have almost 1/10th (~210 or so). This is a
problem while training NNs. A common technique is to _augment the images_ so that network can learn
better (i.e. generalize better) while identifying known images under various conditions (e.g.
darker/brighter images, signs from an angle, etc.)

### 2. Data preprocessing / augmentation

For pre-processing, I used techniques that were explored before, in particular normalizing the image
data, and equalizing the histograms. Images in the wild, in general, are prone to various kinds of
distortions and noise, e.g. dark / bright pixels, highly saturated colors, etc. that can throw off
feature recognition at pixel-levels. Normalizing the image (dividing pixel values by 255 so that they
range from [0, 1] instead of regular [0, 255] and dividing by the mean ensures that all pixels count
equally. Equalizing the image's histogram [see detail here](https://en.wikipedia.org/wiki/Histogram_equalization) ensures that pixel intensities are better distributed throughout the image, thus removing the effect of too-bright or too-dark pixels. This contrast normalization helps the network determine the true features (e.g. curves, edges) within an image rather than false identification of features.

For **augmentation**, I used Keras's built-in **ImageDataGenerator** class. This was a primary reason
for me to consider using Keras, as the built-in capabilities provide a higher-level abstracted API for
    image pre-processing. I used the following augmenation methods:

- Rotation: rotated images by +/- 15 degrees
- Shearing / Shifting: sheared and shifted (horizontally / vertically) by 10% of the image dimensions

Keras also provides automatic _feature-wise mean subtraction, feature-wise normalization_ and various
other pre-processing primitives. However, some of these did not work for me due to OpenCV errors;
additionally, my images were already normalized by 1/255, so I didn't pursue this further.


### 3. Model architecture

Having read some of the papers that implemented deep (and deeper) architectures for image
classification, I was looking to try some simpler architectures, without too much loss in performance.

I started with a very simple LeNet model that yielded roughly 92%-94% accuracy. This was good enough,
but only a baseline. 
Vivek Yadav's architecture (see here) achieves a 98.8% accuracy, however, it is vastly more complex. I
wanted to try out a simpler model. 

After various manual trials without the help of grid search, I settled for a _VGG lite_ style model. 

![png](AA-CNN-model-traffic-signs.jpg)

The model consists of three main blocks:

a) Block 1 has two convolution layers, each with a 3x3 kernel of depth 16, followed by a maxpool of 2x2
b) Block 2 is similar, but has 5x5 kernels of depth 32, followed by a maxpool of 2x2
c) Block 3 is a set of fully connected layers with decreasing number of neurons (1024, 512, 256)
connected to the final layer of 43 neurons that are activated by a softmax function. 

Each block uses Dropout for regularization so that the network maintains redundancy and does not
overfit.

### 4. Training

I trained the model on my Macbook Pro [(which has an Nvidia GPU); however, due to some odd reasons,
despite having CUDA v8 and a TensorFlow library that supports GPU, my GPU didn't work in this case. ] 

I used the following parameters:

- Epochs. I trained the model for ~15 epochs and watched for performance improvement. If the performance
  improved, I further trained for 20-30 epochs.

- Learning Rate Schedule. Starting with LR of 0.01, and decaying at 1e-6. Additionally, I used Keras's
  callback function `ReduceLROnPlateau` which reduces learning rate when certain criteria are met. I
  used a `patience=2` meaning if the performance didn't improve in 2 epochs, the LR would be
  automatically reduced.

- Batch Size. I used a Batch Size of 64 (and sometimes 128), which seemed to work for me. However, a better technique is listed
  [here](http://stats.stackexchange.com/questions/140811/how-large-should-the-batch-size-be-for-stochastic-gradient-descent)

- Optimizer. For optimization, I used the regular _Stochastic Gradient Descent_ optimizer. I also
  experimented with the Adam optimizer, but didn't end up using it.

The total training process took about 30-40 minutes for about 15-20 epochs. Further training (on saved
model) took roughly the same amount of time.

I used Keras's built-in `ModelCheckpoint` callback capability to save the best models while training.


### 5. Model Performance

On the Test dataset (with ~12000 sample images), the model achieved a **98.15 accuracy**. This is not
ground-breaking, but is decently good for such a simple model.

On unseen images (a random collection of US and some German roadsigns), the model achieved less
accuracy. Road signs that were similar to German road signs (e.g. STOP sign, Yield, No U Turn, etc.) were accurately identified (as expected). Others were only corectly identified in the Top5 category. Still others were incorrectly identified.

The details of the model are shown in the included Python notebook.


---


### Overview

In this project, you will use what you've learned about deep neural networks and convolutional neural networks to classify traffic signs. You will train a model so it can decode traffic signs from natural images by using the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). After the model is trained, you will then test your model program on new images of traffic signs you find on the web, or, if you're feeling adventurous pictures of traffic signs you find locally!

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [Pandas](http://pandas.pydata.org/) (Optional)

Run this command at the terminal prompt to install [OpenCV](http://opencv.org/). Useful for image processing:

- `conda install -c https://conda.anaconda.org/menpo opencv3`

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.
2. Clone the project and start the notebook.
```
git clone https://github.com/udacity/CarND-Traffic-Signs
cd CarND-Traffic-Signs
jupyter notebook Traffic_Signs_Recognition.ipynb
```
3. Follow the instructions in the `Traffic_Signs_Recognition.ipynb` notebook.

