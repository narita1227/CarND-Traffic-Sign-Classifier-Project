# **Traffic Sign Recognition** 

[//]: # (Image References)

[image1]: ./others/visualization.png "Visualization"
[image2]: ./others/grayscale.png "Grayscaling"
[image3]: ./others/normalized.png "Normalizing"
[image4]: ./others/img0.png "Traffic Sign 1"
[image5]: ./others/img1.jpg "Traffic Sign 2"
[image6]: ./others/img2.jpg "Traffic Sign 3"
[image7]: ./others/img3.jpg "Traffic Sign 4"
[image8]: ./others/img4.jpg "Traffic Sign 5"

---

### Data Set Summary & Exploration

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the numpy methods to calculate summary statistics of the traffic
signs data set:

* The size of training set is 34799.
* The size of the validation set is 4410.
* The size of test set is 12630.
* The shape of a traffic sign image is (32, 32, 3).
* The number of unique classes/labels in the data set is 43.

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. It is a bar chart showing how training data, validation data, test data are distributed.

![alt text][image1]

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data.

As a first step, I decided to resize the images to 32x32 because the new images from the web are not necessarily so.

As a second step, I converted the images to grayscale because the accuracy was better when not using color than when using color.
Here is an example of a traffic sign image before and after grayscaling.

![alt text][image2]

As a last step, I normalized the image data to suppress variations in the brightness of the image.
Here is an example of a traffic sign image before and after normalizing.

![alt text][image3]


#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.

My final model consisted of the following layers:

| Layer         	|     Description	        		|
|:---------------------:|:---------------------------------------------:|
| Input         	| 32x32x1 grayscale image   			|
| Convolution 5x5     	| 1x1 stride, no padding, outputs 28x28x6 	|
| RELU			| activation					|
| Max pooling	      	| 2x2 stride, outputs 14x14x6 			|
| Convolution 5x5     	| 1x1 stride, no padding, outputs 10x10x16 	|
| RELU			| activation					|
| Max pooling	      	| 2x2 stride, outputs 5x5x16 			|
| Flatten	     	| outputs 400				 	|
| Fully Connection	| outputs 120					|
| RELU			| activation					|
| Dropout		| keep_prob is 0.5				|
| Fully Connection	| outputs 84					|
| RELU			| activation					|
| Fully Connection	| outputs 43					|
| Softmax		| activation					|

I used LeNet almost as it was. I added a dropout layer after the first fully-connected layer to prevent overfitting.


#### 3. Describe how you trained your model.

To train the model, I used the following parameters.

| Parameters         	|     Description	        		|
|:---------------------:|:---------------------------------------------:|
|Type of Optimizer	| Adam						|
|Batch Size		| 128						|
|Number of Epochs	| 40						|
|Learning Rate		| 0.001						|

Batch Size and Learning Rate are set according to the sample shown in the lesson.
Even if Number of Epochs was increased more than 40, the accuracy did not rise, so I set it to 40.


#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93.

My final model results were:
* training set accuracy of 0.999.
* validation set accuracy of 0.956.
* test set accuracy of 0.945.

I chose LeNet-5 architecture because it is similar to the architecture shown in the paper on recognizing traffic signs.
At first, I used LeNet-5 as it was. The accuracy on the training set was pretty good, but the accuracy on the validation set was low.
So I think overfitting is occurring. Then, I added a dropout layer after the first fully-connected layer to prevent overfitting, and increased the number of epochs.
As a result, the accuracy is good not only for training set but also for validation set and test set, I think the model is working well.

### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report.

Here are five German traffic signs that I found on the web:

![alt text][image4] ![alt text][image5] ![alt text][image6] 
![alt text][image7] ![alt text][image8]

The first and second images look relatively easy to classify, so I used these images for sanity check.
The third and fourth images might be difficult to classify because they are a bit cloudy and dark.
The fifth image might be difficult to classify because the sign is taken from an oblique angle.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set.

Here are the results of the prediction:

| Image					|     Prediction	        		|
|:-------------------------------------:|:---------------------------------------------:|
| Speed limit (60km/h)  		| Speed limit (60km/h)   			|
| Stop     				| Stop 						|
| Priority road				| Priority road					|
| Right-of-way at the next intersection	| Right-of-way at the next intersection		|
| Speed limit (50km/h)			| Speed limit (50km/h)      			|


The model was able to correctly guess 5 of the 5 traffic signs, which gives an accuracy of 100%.
This compares favorably to the accuracy on the test set of 94.5%.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction.

The code for making predictions on my final model is located in the 16th cell of the Ipython notebook.

For the first image, the model is relatively sure that this is a Speed limit (60km/h) (probability of 0.99), and the image does contain a Speed limit (60km/h).
The top five soft max probabilities were

| Probability         	|     Prediction	     				|
|:---------------------:|:-----------------------------------------------------:|
| 0.99		   	| Speed limit (60km/h)   				|
| 3.4e-3     		| Speed limit (80km/h) 					|
| 6.5e-4		| Speed limit (50km/h)					|
| 1.2e-5      		| End of no passing by vehicles over 3.5 metric tons	|
| 6.4e-7		| End of speed limit (80km/h)      			|

For the second image, the model is relatively sure that this is a Stop (probability of 0.856), and the image does contain a Stop.
The top five soft max probabilities were

| Probability         	|     Prediction	     				|
|:---------------------:|:-----------------------------------------------------:|
| 0.856		   	| Stop   						|
| 0.136     		| Bumpy road 						|
| 4.2e-3		| Turn left ahead					|
| 2.4e-3      		| Yield							|
| 6.7e-4		| Keep right      					|

For the third image, the model is relatively sure that this is a Priority road (probability of 0.99), and the image does contain a Priority road.
The top five soft max probabilities were

| Probability         	|     Prediction	     				|
|:---------------------:|:-----------------------------------------------------:|
| 0.99		   	| Priority road   					|
| 2.1e-7     		| No vehicles 						|
| 1.5e-11		| No passing						|
| 6.4e-13      		| End of no passing					|
| 3.3e-13		| End of all speed and passing limits      		|

For the forth image, the model is relatively sure that this is a Right-of-way at the next intersection (probability of 1.00), and the image does contain a Right-of-way at the next intersection.
The top five soft max probabilities were

| Probability         	|     Prediction	     				|
|:---------------------:|:-----------------------------------------------------:|
| 1.00		   	| Right-of-way at the next intersection  		|
| 3.5e-14     		| Beware of ice/snow 					|
| 1.1e-14		| Roundabout mandatory					|
| 1.4e-19      		| Pedestrians						|
| 1.3e-19		| Children crossing      				|

For the fifth image, the model is relatively sure that this is a Speed limit (50km/h) (probability of 0.99), and the image does contain a Speed limit (50km/h).
The top five soft max probabilities were

| Probability         	|     Prediction	     				|
|:---------------------:|:-----------------------------------------------------:|
| 0.99		   	| Speed limit (50km/h)  				|
| 5.7e-3     		| Wild animals crossing 				|
| 2.1e-4		| Speed limit (60km/h)					|
| 4.9e-5      		| Speed limit (30km/h)					|
| 4.8e-8		| Dangerous curve to the left      			|

### (Optional) Visualizing the Neural Network
#### 1. Discuss the visual output of your trained network's feature maps.
The shape of the circle of the traffic sign is recognized.
