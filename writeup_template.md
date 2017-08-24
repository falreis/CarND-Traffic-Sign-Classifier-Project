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
[noise_reduction]: ./ewriteup_images/noise_reduction.png "Image before and after the noise reduction"
[image0]: ./german-traffic-signs/00000.ppm "German Traffic Sign Image 0"
[image1]: ./german-traffic-signs/00001.ppm "German Traffic Sign Image 1"
[image2]: ./german-traffic-signs/00002.ppm "German Traffic Sign Image 2"
[image3]: ./german-traffic-signs/00003.ppm "German Traffic Sign Image 3"
[image4]: ./german-traffic-signs/00004.ppm "German Traffic Sign Image 4"
[image5]: ./german-traffic-signs/00005.ppm "German Traffic Sign Image 5"
[image6]: ./german-traffic-signs/00006.ppm "German Traffic Sign Image 6"
[image7]: ./german-traffic-signs/00007.ppm "German Traffic Sign Image 7"
[image8]: ./german-traffic-signs/00008.ppm "German Traffic Sign Image 8"
[image9]: ./german-traffic-signs/00009.ppm "German Traffic Sign Image 9"


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

As a first step, I decided to reduce the noise, calculating the average pixel data for the all image and then reducing it for each pixel.

I also turn the image into grayscale, but the image started to lost the contour of the important data. Then I stepped back and only apply the noise reduction.

Here is an example of a traffic sign image before and after the noise reduction.

![Image before and after the noise reduction][noise_reduction]

#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

##### Layer 1
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, same padding, outputs 28x28x6 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 14x14x6 			     	|

##### Layer 2
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 14x14x6 RGB image   							| 
| Convolution 5x5    	| 1x1 stride, same padding, outputs 10x10x16	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 5x5x16 			     	|
| Flatten		      	| Input = 5x5x16. Output = 400.			     	|

##### Layer 3
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 400   										| 
| MatMul + RELU	    	| 												|
| Output		      	| 120									     	|

##### Layer 4
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 120   										| 
| MatMul + RELU	    	| 												|
| Output		      	| 84									     	|

##### Layer 5
| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 84	   										| 
| MatMul + RELU	    	| 												|
| Output		      	| 43									     	|

#### 3. Describe how you trained your model. The discussion can include the type of optimizer, the batch size, number of epochs and any hyperparameters such as learning rate.

To train the model, I used an ....

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of 0.941
* test set accuracy of 0.920

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* What were some problems with the initial architecture?
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* Which parameters were tuned? How were they adjusted and why?
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?

If a well known architecture was chosen:
* What architecture was chosen?
* Why did you believe it would be relevant to the traffic sign application?
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
 
---
### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are ten German traffic signs that I found on the web:

![German Traffic Sign Image 0][image0] ![German Traffic Sign Image 1][image1] ![German Traffic Sign Image 2][image2] ![German Traffic Sign Image 3][image3]  ![German Traffic Sign Image 4][image4] 
![German Traffic Sign Image 5][image5] ![German Traffic Sign Image 6][image6] ![German Traffic Sign Image 7][image7] ![German Traffic Sign Image 8][image8] ![German Traffic Sign Image 9][image9]


1. The first image maybe be simple to classify because it's a good picture of the traffic sign;
2. The second image might be difficult to classify because it's not a plain image, the sign seens to be warped;
3. The third image probably will be easily classified;
4. The fourth image has some problems like the background with different colors that can  confuse the algorithm;
5. The fifth image probably will be easily classified;
6. The sixth image probably will be easily classified. The image has a red color in the background but I think that will be not enought to confuse the algorithm;
7. The seventh image can confuse the algorithm because it has a white background and it's a white sign;
8. The eithgh image has bad resolution and colors and can be difficult to classify;
9. The nineth image can be easily classified;
10. The tenth image has some tree branchs that can confuse the algorithm. Also, the image seens to has some kind of shadow, that can be difficult to ignore.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set (OPTIONAL: Discuss the results in more detail as described in the "Stand Out Suggestions" part of the rubric).

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Stop Sign      		| Stop sign   									| 
| U-turn     			| U-turn 										|
| Yield					| Yield											|
| 100 km/h	      		| Bumpy Road					 				|
| Slippery Road			| Slippery Road      							|


The model was able to correctly guess 4 of the 5 traffic signs, which gives an accuracy of 80%. This compares favorably to the accuracy on the test set of ...

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. (OPTIONAL: as described in the "Stand Out Suggestions" part of the rubric, visualizations can also be provided such as bar charts)

The code for making predictions on my final model is located in the 11th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a stop sign (probability of 0.6), and the image does contain a stop sign. The top five soft max probabilities were

| Probability         	|     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| .60         			| Stop sign   									| 
| .20     				| U-turn 										|
| .05					| Yield											|
| .04	      			| Bumpy Road					 				|
| .01				    | Slippery Road      							|


For the second image ... 

---
### (Optional) Visualizing the Neural Network (See Step 4 of the Ipython notebook for more details)

#### 1. Discuss the visual output of your trained network's feature maps. What characteristics did the neural network use to make classifications?
