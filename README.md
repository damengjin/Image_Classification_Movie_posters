# Image Classification of Movie Posters

## 1. Summary of the project

The purpose of this project is to proceed a binary classification on the movie posters (extracted from IMDB), and to learn whether there is any correlation between whether the movie is fresh or rotten and how the posters look. If so, there could be more insights provided to the movie producers for making decisions on movie posters. Transfer learning has been adopted and performed on the google collabtory platform with GPU, in order to handle big calculation. The details will be briefly presented in this Readme file:

## 2. Introduction and pre-processing on the images:

From the dataset, more than 16K valid posters were provided with each poster of size (305, 206, 3). Every poster was then resized to be (224, 224, 3) for the convenience of applying neural networks in the later sessions. Some features of the posters were explored by denoise to separate background from foreground or identify different objects with marked label. These resized posters were then transformed into a large numpy-array (16000, 224, 224, 3), which, together with their corresponding labels of the movie being either fresh or rotten were passed to the deep learning models to be trained. This image classification task was objective to discover the relationship between movie posters and the probability of the movie being fresh, hopefully, providing insights for movie producers on the poster design.

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/1.png)
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/2.png)
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/3.png)

## 3. Techniques used for boosting CNN performance:

1). Data Argumentation:

Image argumentation strategy to boost the performance of deep learning networks due to two main reasons: one is that deep neural networks require a large amount of training data to prevent overfitting, the other one is that the orientation of the image could affect the model performance. Therefore, the image augmentation technique was applied here to artificially creates training images through different ways of processing or combination of multiple processing, such as random rotation, shifting, shearing, and flipping of the input posters. Take the poster of “Sun choke” as an example, with Keras’ ImageDataGenerator API, following augmented images were generated:

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/4.png)

2). Transfer Learning: 

In transfer learning, one repurposes the learned features, or transfer the knowledge from a relevant trained task to a second target network wait to be trained. Some pre-trained image networks are VGG-16 from Oxford, and Inceptions from Google. Take the VGG-16 as an example (with its structure illustrated in figure xxx). It contains 13 layers: 5 blocks of convolution layers each with a max-pooling layer for down-sampling; 2 fully connected dense layers, and 1 output layer containing 1000 classes. For transfer learning, the last three layers of the VGG-16 will be dropped, since we would use our own fully connected dense layers to do the binary classification on whether the posters will be fresh or rotten with “sigmoid” as activation function. 

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/5.png)

One strategy is to use VGG-16 as feature extraction tool, where all blocks weight will be fixed (non-trainable). While the mostly used strategy is to replace and retrain the classifier on top of the pre-trained networks on the new dataset and also to fine-tune the weights of the pre-trained network by continuing the backpropagation. It is efficient to only fine-tune some high-level portion of the network while keep earlier layers fixed. Because it is discovered in previous research (and later demonstrated in the next session) that the earlier features of a pre-trained network learn more generic features, whereas the deeper layers of the network become progressively more specific to the details of the classes contained in the original dataset. With this strategy, the network is more capable of recognizing the poster patterns. The demonstration of both strategies using VGG-16 pre-trained network is shown in the figure.

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/6.png)

## 4. Models and Results:
Models	Training accuracy	Validation accuracy
Basic CNN	0.99	0.55
CNN with Regularization	0.75	0.56
Regularized CNN with Image Argumentation	0.59	0.58
Transfer Learning VGG-16 (Feature Extractor)	0.62	0.61
Transfer Learning VGG-16 (Feature Extractor) with Argumentation	0.62	0.61
Transfer Learning VGG-16 with Fine-Tuning and Argumentation	0.62	0.62
Transfer Learning InceptionResNetV2 with Fine-Tuning and Argumentation	0.66	0.62
Transfer Learning InceptionV3 with Fine-Tuning and Argumentation	0.64	0.61

The Accuracy and Loss Plot for each model is presented below:
The x-axis is epoch number, we can see that for transfer learning models, the learning has been picked up faster, and the accuracy rised up for a short number of epochs:

### Basic CNN
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/7.png)

### Basic CNN with Regularization
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/8.png)

### Regularized CNN with image augmentation
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/9.png)

### Transfer Learning VGG-16 (Feature Extractor) 
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/10.png)

### Transfer Learning VGG-16 (Feature Extractor) with image augmentation
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/11.png)

### Transfer Learning InceptionResNetV2 (fine-tuned) with image augmentation
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/12.png)

### Final chosen model: Transfer Learning VGG-16 (fine-tuned) with image augmentation
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/13.png)

A basic CNN model with three convolutional layers, coupled with max pooling for auto-extraction of features from the poster images and also down-sampling the output convolution feature maps. From the result, it can be seen that the model is overfitting after 5-7 epochs even though there are 7,680 training posters. Then another convolution layer with dense hidden layer was added to the basic CNN, with dropout of 0.3 after each hidden layer to enable regularization. Dropout randomly masks the 30% of units from the hidden dense layers and set the outputs to 0. However, the results still ended up overfitting around 75% (better than 99% in basic CNN, but still not good enough). Thereafter, image augmentation strategy was applied with zooming, rotating, shifting, shearing, and flipping to the existing posters and fed them to the CNN. This quite improved the overfitting issue, and the accuracy also jumps from 55% to 58.5% (can be improved). Next step, pre-trained CNN models were leveraged to further boost up the performance with transfer learning. There were three different pre-trained models used in this project. They were VGG-16, InceptionResNetV2, and InceptionV3. The VGG-16 was initially used as a simple feature extractor by freezing all the five convolution blocks to make sure their weights were not updated after each epoch and only train the model on two more dense layers added after the VGG-16 pre-trained model. For the second variant, last two blocks of VGG-16 model were unfrozen (Block 4, and 5) and their weights were getting updated in each epoch as the model was trained. The results showed that with fine tuning, the best model was obtained with validation accuracy boosted up to 62%. Similarly, pre-trained model InceptionResNetV2 and InceptionV3 were also used with fine tuning on the convolution blocks, the validation accuracy of these models did not outperform VGG-16 (Even though training accuracy while using InceptionResNetV2 was boosted up to 66% due to overfitting a bit). Therefore, fine-tuning transfer learning using VGG-16 was used as our final model. The model was tested on the untouched 3200 posters, the test accuracy was 62.3% as the best performance, and the classification report with ROC plot were shown below:

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/14.png)
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/15.png)

The accuracy of 62.3% for a binary classification problem was not high. To better interpret the result, intermediate layers and CAM techniques would be applied afterwards to discuss the contribution of the image classification task to the achievement of the whole project.


## 5. Visualization of Intermediate layers:

The pre-trained deep CNN models used for transfer learning, like VGG-16 (visual geometry group) and inception architecture, are meant for achieving more effective feature extraction using the existing acknowledge with less data availability. To better understand the pre-trained model on how it is able to classify the input image, it is useful to look at the output of its intermediate layers. Take the InceptionV3 architecture as an example, with a poster input, 3 convolution and their activation layers have been shown below:  

### input original poster
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/16.png)

### Filters from first convolution layer
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/17.png)

### Filters from ReLu Activation layer for first convolution layer
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/18.png)

### Filters from fourth convolution layer
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/19.png)

### Filters from ReLu Activation layer from fourth convolution layer
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/20.png)

### Filters from ninth convolution layer
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/21.png)

### Filters from ReLu Activation layer from ninth convolution layer 
![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/22.png)

With the help of these visualizations of the intermediate layer, it is pretty clear to see how different filters in different convolution layers are trying to highlight or activate different parts of the input image. Some act as edge detectors, others detect a particular region of the poster like the title, darker colored portion, or background. It is easier to see this behavior of convolution layers in the starting layers (more general patterns), since as model went deeper the pattern captured by the convolution kernel become more and more sparse. Deeper the layers in the network, more training data specific features were visualized. In the next section, another visualization technique, Class Activation Maps (CAM) was used to better interpret the result of the final output.


## 6. Class activation map & Result interpretation:

Class Activation Maps (CAM) technique is widely used when interpretation of the output is crucial in the use case. For the poster classification, since the objective is to provide movie producers insightful information on the design of posters, CAM was an essential step of the project delivery.  A class activation map for a particular category indicates the discriminative region used by CNN to identify the category. This is achieved by projecting back the weights of the output layer on the convolution feature maps obtained from the last convolution layer (GAP layer, which takes an average cross all the activation to find all the discriminative regions). 

With the 0.623 test accuracy on the 3200 test posters, we sort them by the output probability of being fresh to obtain the top 8 (most likely fresh) and bottom 8 (most likely rotten) posters. Then apply CAM on these posters see what CNN have learned to predict the posters.  

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/Screenshot%202020-11-27%20at%2010.34.47%20PM.png)

![alt text](https://github.com/damengjin/Image_Classification_Movie_posters/blob/main/img/Screenshot%202020-11-27%20at%2010.35.00%20PM.png)

From the result, without discuss the accuracy of the prediction, we can see clearly how the final model predict the posters using heatmap. The heatmap shows the score (weight) that the location of the image possesses in predicting the output class of the image. Since this is a binary classification using “sigmoid” function as the output layer, we can focus on the warm region for fresh class, and cold one for rotten class. For the top posters that are predicted fresh (with probability > 0.9), they all have very big title (some titles are stacking) occupying most of the poster content space; and there are very few characters in the poster. For those predicted to be rotten (p < 0.2), the posters tend to have a lot of characters filling up almost the whole poster, and most of these characters are standing in parallel or regular patterns. 

In terms of accuracy, some of these top predicted posters were mis-classified, since the prediction model only has an accuracy of 0.623 on the test posters. There are some limitations of the final image classification model: First is the title in the poster. Titles are supposed to be part of the poster and it is the purpose of delivering what kind of font or layout are more positively catching audience’s eyeballs. However, the model now seemed to decide as long as there are plenty of large font of texts inside, then it is fresh. Secondly, the accuracy is relatively low could be due to the effect of movie genre or year of release. The styles of posters could be similar and of mainstream within a period of time or of a certain genre. And a comedy poster tends to be fresh could look extremely different from an attractive thriller poster. Third point is that besides the two factors discussed in the second point, there are so many other factors that affect the feedback of the movie: like the ones we analyzed in the prediction analysis section, like directors, casts, and most importantly the storytelling. The goal of the poster analysis is not giving director absolute insights barely from image, but to provide additional useful tips before the movie production to help them make better decisions.
