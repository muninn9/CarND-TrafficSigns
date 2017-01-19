# CarND-TrafficSigns

<b>ANSWER 1</b><br>
<i>Describe how you preprocessed the data. Why did you choose that technique?</i><br><br>
I cannot "provide a good concise explanation for the preprocessing techniques used" because I did not preprocess the data.  This is because all of the preprocessing techniques I tried did not improve the accuracy. However, I can provide a good concise explanation for the preprocessing techniques that are commonly used and that I tried to use. 
I tried subtracting the mean from the images to zero center them by doing this:<br><br> 

&nbsp;&nbsp;&nbsp;&nbsp; X_train = np.subtract(X_train, np.mean(X_train, axis=0)) <br><br>

but it did not improve the accuracy. This works by computing the mean value of every pixel in the training set and subtracting it out. If I did use this it would be because it usually standardizes the data. Because it centers all of the data points of each image around the origin, it puts all the images on a level playing field thus making it easier for the classifier to learn. It is also common to do per-channel mean subtraction. This works by computing the mean of each red, blue, and green channel which yields 3 numbers representing the mean of each channel. These are then subtracted out.  I found that another common preprocessing technique is to divide the mean by the standard deviation but because all the pixel values are in the same range this was not necessary. Other common preprocessing techniques are PCA and whitening but they are not usually used with images.

<br><br><b>ANSWER 2</b><br>
<i>Describe how you set up the training, validation and testing data for your model. Optional: If you generated additional data, how did you generate the data? Why did you generate the data? What are the differences in the new dataset (with generated data) from the original dataset?</i><br><br>
The train, validation, and test sets were created using the following line of code:<br><br>

&nbsp;&nbsp;&nbsp;&nbsp; X_train, X_validation, y_train, y_validation = train_test_split(X_train, y_train, test_size=0.20, random_state=0)

<br><br><b>ANSWER 3</b><br>
<i>What does your final architecture look like? (Type of model, layers, sizes, connectivity, etc.) For reference on how to build a deep neural network using TensorFlow, see Deep Neural Network in TensorFlow from the classroom.</i><br><br>
I used a LeNet model with two layers and filter sizes of 5X5. There are three fully connected layers. I ended up using both average pooling and max pooling which improved the accuracy. I also added dropout regularization to the convolutional layers which improved accuracy.

<br><br><b>ANSWER 4</b><br>
<i>How did you train your model? (Type of optimizer, batch size, epochs, hyperparameters, etc.)</i><br><br>
The type of optimizer I ended up using was the Adam optimizer. This because it has been proven to be the best in current research. After adding dropout, I tried higher epoch values and 30 ended up giving me satisfactory accuracy. I think this is because dropout creates sparser features so it takes longer to learn the images. Since it can't rely on particular features it must figure out alternative ways or paths of understanding the image. The benefit of this is that it gives a more complete and resilient understanding of the images.

<br><br><b>ANSWER 5</b><br>
<i>What approach did you take in coming up with a solution to this problem? It may have been a process of trial and error, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think this is suitable for the current problem.</i><br><br>
I began with the LeNet architecture. This is what was recommended and it was giving pretty good accuracy. Then through trial and error I tried modifying it with regularization, normalization, and different types of pooling. First I tried adding average pooling in addition to max pooling as current research has proven that to be most effective. It did improve accuracy. Then, I determined that I must have been overfitting because I had high validation accuracy but lower test accuracy. So in this case regularization made sense. That is why I added drop out. It improved accuracy but I was still overfitting due to a high validation accuracy and lower test accuracy. So, I applied L2 regularization and this also helped a little. However, I still think I'm overfitting and I think perhaps augmenting the data could improve the test accuracy because the model wouldn't be so specific and it would become more generalized which would allow a comprehension of new inputs.

<br><br><b>ANSWER 6</b><br>
<i>Choose five candidate images of traffic signs and provide them in the report. Are there any particular qualities of the image(s) that might make classification difficult? It could be helpful to plot the images in the notebook.</i><br><br>
This code is in 'predict.py'. It classifies all the images correctly.

<br><br><b>ANSWER 7</b><br>
<i>Is your model able to perform equally well on captured pictures when compared to testing on the dataset? The simplest way to do this check the accuracy of the predictions. For example, if the model predicted 1 out of 5 signs correctly, it's 20% accurate.</i><br><br>
Based on the 5 input signs the accuracy was 100%. I believe with a bigger dataset it would be comparable to the test accuracy.

<br><br><b>ANSWER 8</b><br>
<i>Use the model's softmax probabilities to visualize the certainty of its predictions, tf.nn.top_k could prove helpful here. Which predictions is the model certain of? Uncertain? If the model was incorrect in its initial prediction, does the correct prediction appear in the top k? (k should be 5 at most)</i><br><br>
This is also in 'predict.py'. The correct prediction always appears in the top 3.
