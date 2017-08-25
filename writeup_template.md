**Traffic Sign Recognition**
---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[csv_classes]: ./writeup_images/csv_classes.png "CSV classes"
[train_percentages]: ./writeup_images/train_percentages.png "Classes and percentages"
[pie_graph]: ./writeup_images/pie_graph.png "Pie graph with classes and percentages"
[noise_reduction]: ./writeup_images/noise_reduction.png "Image before and after the noise reduction"
[image0]: ./german-traffic-signs/00000.jpg "German Traffic Sign Image 0"
[image1]: ./german-traffic-signs/00001.jpg "German Traffic Sign Image 1"
[image2]: ./german-traffic-signs/00002.jpg "German Traffic Sign Image 2"
[image3]: ./german-traffic-signs/00003.jpg "German Traffic Sign Image 3"
[image4]: ./german-traffic-signs/00004.jpg "German Traffic Sign Image 4"


## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! and here is a link to my [project code](https://github.com/falreis/CarND-Traffic-Sign-Classifier-Project/blob/master/Traffic_Sign_Classifier.ipynb)

---
### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library and numpy to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799
* The size of test set is 12630
* The shape of a traffic sign image is (32, 32, 3)
* The number of unique classes/labels in the data set is 43

#### 2. Include an exploratory visualization of the dataset.

To visualize the data set I printed some informations. 
Firstly, I printed all the classes in the CSV file and the names of the classes.

![CSV classes][csv_classes]

Then, I printed the classes names, numbers and the percentage of information of each class.

![Classes and percentages][train_percentages]

To finish, I plotted a pie graph with the percentages of each class in the train set.

![Pie graph with classes and percentages][pie_graph]

---
### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

As a first step, I decided to reduce the noise, calculating the average pixel data for the all image and then reducing it for each pixel. I also divide the subtraction by the average, normalizing the image. The formula used was:

*pixel = (pixel - avg(image)) / avg(image)*

I also turn the image into grayscale, but the image started to lost the contour of the important data. Then I stepped back and only apply the normalization.

Here is an example of a traffic sign image before and after the normalization.

![Image before and after the noise reduction][noise_reduction]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

##### Layer 1
| Layer | Description | 
|:-----:|:-----------:| 
| Input | 32x32x3 RGB image | 
| Convolution 5x5 | 1x1 stride, same padding, outputs 28x28x6 |
| RELU | |
| Max pooling | 2x2 stride,  outputs 14x14x6 |

##### Layer 2
| Layer | Description | 
|:-----:|:-----------:| 
| Input | 14x14x6 RGB image | 
| Convolution 5x5 | 1x1 stride, same padding, outputs 10x10x16 |
| RELU | |
| Max pooling | 2x2 stride, outputs 5x5x16 |
| Flatten | Input = 5x5x16. Output = 400. |

##### Layer 3
| Layer | Description | 
|:-----:|:-----------:| 
| Input | 400 | 
| MatMul + RELU | |
| Output | 120 |

##### Layer 4
| Layer | Description | 
|:-----:|:-----------:| 
| Input | 120 | 
| MatMul + RELU | |
| Output | 84 |

##### Layer 5
| Layer | Description | 
|:-----:|:-----------:| 
| Input | 84 | 
| MatMul + RELU | |
| Output | 43 |


#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I kept the LeNet configuration, with few changes to accept images with 3 channels. I also kept the same architecture, with AdamOptimizer train optimizer. Also, I kept the minimize to the loss operation.

The following parameters that I used was.

| Parameter | Value | 
|:---------:|:-----:| 
| epochs | 20 | 
| batch_size | 128 |
| mu (conv. weight) | 0 |
| sigma (conv. weight) | 0.03 |
| rate | 0.0025	|

To define the parameters I started with the LeNet MNIST project parameters. 

The I started to refine the model and increase or decrease the parameters to get the best result, not overtraining or undertraining the solution.

The parameters that I described in the table above, is a good solution found for the parameters. Sometimes. using 15 epochs, the results is same as to the 20 epochs solution. I prefered to use 20 epochs to keep the results more stable.

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of **0.938**
* test set accuracy of **0.916**

The interative aproaching used explanation:

**1. What was the first architecture that was tried and why was it chosen?**

The first LeNET architecture used was the same that I used in the end of the project. The changes were made in the normalized function. As discussed before, I used firsty the aproximation with *(pixel-128) / 128*, but the result was not good as expected. Then I also tried to convert image into grayscale, but the results not improved so much and I also had to change my architecture, because grayscale use only 1 channel.
Then I thought to use the average to normalize the results. This option lead to improve my performance.

**2. What were some problems with the initial architecture?**

The problems with the initial architecture was the limitation of recognition. The initial architecture never has accuracy over 0.89, even if I tune the parameters to the best performance.

**3. How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.**

The architecture was kept the same. The ajustment was only to tune parameters to have the best performance. 

**4. Which parameters were tuned? How were they adjusted and why?**

The parameters tuned was epochs, rate and sigma. The epochs parameter was tuned because sometimes the result not converge as expected. Rate was tuned to learn in the best way possible. Sigma was used in the iterative approach to increase the results.

**5. What architecture was chosen?**

The LeNET architecture as chosen.

**6. Why did you believe it would be relevant to the traffic sign application?**

The architecture needs some tunes to be even more relevant in the traffic sign real application.

**7. How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?**

The evidence of accuracy was the results over 90% using the images from german sign database. The database has some difficult images to identify the results, as expected in the streets.
 
---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![German Traffic Sign Image 0][image0] ![German Traffic Sign Image 1][image1] 
![German Traffic Sign Image 2][image2] ![German Traffic Sign Image 3][image3] 
![German Traffic Sign Image 4][image4] 

The images found on the internet maybe be easily classified because of its quality and good resolution. Some can be a little difficult, because of the background, but the training set has some more difficult information that can be more difficult to be classified then the images found on the web.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image No. | Image | Prediction	| 
|:--------:|:-----:|:----------:| 
| (16) | Vehicles over 3.5 metric tons prohibited | Vehicles over 3.5 metric tons prohibited| 
| (01) | Speed limit (30km/h)	| Speed limit (30km/h)|
| (38) | Keep right	| Keep right|
| (33) | Turn right ahead	| Turn right ahead|
| (11) | Right-of-way at the next intersection	| Right-of-way at the next intersection|

The model was able to correctly guess 5 of 5 traffic signs, which gives an accuracy of 100%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

<table>
<th>Probability</th><th>Image</th>
<tr>
<td> 
[ 0.60746935  0.60746935  0.60746935]<br>
[ 0.60746935  0.60746935  0.60746935]<br>
[ 0.60746935  0.60746935  0.60746935]<br>
[ 0.60746935  0.60746935  0.60746935]<br>
[ 0.58855794  0.58855794  0.58855794]
</td>
<td>Vehicles over 3.5 metric tons prohibited</td>
</tr>
<tr>
<td> 
[-0.06427948 -0.19232545 -0.64541117]<br>
[ 0.24106089  0.06376648 -0.55676397]<br>
[ 0.38880624  0.21151182 -0.47796645]<br>
[ 0.31000872  0.15241369 -0.51736521]<br>
[ 0.3297081   0.16226338 -0.63556148]
</td>
<td>Speed limit (30km/h)</td>
</tr>
<tr>
<td>
[-0.1498155  -0.24090669 -0.66599895]<br>
[ 0.01212441 -0.11945177 -0.62551397]<br>
[-0.03848181 -0.21054296 -0.77733263]<br>
[-0.60527148 -0.64575646 -0.93927254]<br>
[-0.39272536 -0.42308909 -0.82793885]
</td>
<td>Keep right</td>
</tr>
<tr>
<td> 
[ 0.54077944  0.54077944  0.52869489]<br>
[ 0.54077944  0.54077944  0.52869489]<br>
[ 0.54077944  0.54077944  0.52869489]<br>
[ 0.54077944  0.54077944  0.52265262]<br>
[ 0.52265262  0.51056808  0.5045258 ]
</td>
<td>Turn right ahead</td>
</tr>
<tr>
<td> 
[ 0.25789453  0.17212899  0.08636346]<br>
[ 0.2507474   0.16498186  0.07921633]<br>
[ 0.23645314  0.16498186  0.08636346]<br>
[ 0.22215889  0.17212899  0.10780484]<br>
[ 0.22215889  0.17927612  0.11495197]
</td>
<td>Right-of-way at the next intersection</td>
</tr>
</table>

---
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
