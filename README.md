# Go Bananas!

by Julia Chapman

[![Bananas!](graphics/keep-calm-and-go-bananas-21.png)](http://jalzymedicine.blogspot.com/2015/)

# Overview
Data Source: Scraped from certain subreddits and google images using:

[Reddit Fetch](https://github.com/nobodyme/reddit-fetch)

[Google Images Download](https://github.com/hardikvasa/google-images-download)

In 2005, an internet meme was started when a woman, trying to sell a TV, put a banana in her sale ad as a unit of measurement. Since then, ['Banana For Scale'](https://knowyourmeme.com/memes/banana-for-scale) has grown in popularity and has been dubbed the ['yardstick of the internet.'](https://www.dailydot.com/unclick/banana-for-scale-meme-history/)

[![meme_origin1](graphics/meme1.jpg)](https://knowyourmeme.com/memes/banana-for-scale)

"Banana for Scale" has also been adapted to other objects. In this case, a double mattress!

[![meme_origin2](graphics/meme2.jpg)](https://knowyourmeme.com/memes/banana-for-scale)

So this got me thinking, what if you could use a banana as an actual unit for scale in an image?

Go to my [Capstone-3](https://github.com/jchapman3773/Capstone-3) to find out!

In the meantime, I set out to create a model that could predict if an image contained a banana, a person, both, or neither.

I also setup a private server on my home desktop with a NVIDIA 1060 3GB GPU to speed things up a bit. This turned out to help a lot, as my final model only took about 30s per epoch to train. This gave me a lot more freedom to mess around with optimizing parameters as my models took just a matter of minutes to train, versus what could have been a matter of hours.

# Data

After scraping Reddit and Google for images, I had about 1,100 images total. I manually filtered the images into 4 categories: Banana, Person, Both, or Neither. Even manually, it was sometimes hard to classify certain images that only had a part of an object in it, like a person's foot or arm. Three of the classes were close to balanced, with 'Both' having double the images of the others. I accounted for this imbalance by calulating sample weights to use in the fitting of my models. I split my data into training, validation, and holdout datasets with splits of 0.65/0.15/0.20 respectively. The training data was also augmented using Keras DataImageGenerator.

Mean Img Size: 1132 X 1243

Stdev: 1043 X 1043

My images varied quit a lot and had a lot of noise.

![banana_stand](graphics/banana_stand.jpg)

# Model

I first started off with a simple CNN. My final simple model used a pattern of Convolution2D and MaxPooling2D layers three times. After those six layers, the model was flattened into a dense layer with a final dense layer with n_categories nodes. There were also dropouts between each layer to help reduce overfitting. The pool size was (2,2) and there were 128 filters in each convolution with the input image size as 300 X 300. The learning rate was 0.00005.

![acc](graphics/Simple_CNN_acc_hist.png)
![loss](graphics/Simple_CNN_loss_hist.png)

Holdout Loss: 0.888

Holdout Accuracy: 0.677

My simple CNN didn't perform very well because I have a very limited set of data with a lot of noise. To help it better learn the features, a few orders of magnitude greater of data would be optimal.

Because of my limited data set, I next took advantage of transfer learning to help my model along. Transfer learning boosts small datasets by loading in initial weights and layers pretraining on large datasets. With the more generalized features, the transfer learning was able to give my model prior information to initiate with.

I used the keras Xception model trained on ImageNet as my initial model. 

[![Xception](graphics/imagenet_xception_flow.png)](https://www.pyimagesearch.com/2017/03/20/imagenet-vggnet-resnet-inception-xception-keras/)

To use the Xception model, I removed the head and added a layer of my own. I added a GlobalAveragePooling2D layer with a Dense layer with n_categories nodes for output.
To retrain the Xception model, I first started with 5 warmup epochs on just the new head (lr=0.0005) with all other layers frozen.
After the warmup, I unfroze the next 6 layers and continued to train (lr=0.00005) for 15 epochs, saving the best model based on validation loss. The input images size was 400 X 400.

![acc](graphics/Transfer_CNN_acc_hist.png)
![loss](graphics/Transfer_CNN_loss_hist.png)

![confusion_matrix](graphics/Confusion_Matrix_with_weights.png)

### Missed Targets

[![fail](graphics/fail.jpg)](https://bized.aacsb.edu/articles/2017/11/why-its-fine-to-fail)

![failed_images](graphics/failed_images.png)

# Results

**Final Model**

Holdout Loss: 0.270

Holdout Accuracy: 0.903

Classification Report:

```
             precision    recall  f1-score   support

     Banana       0.87      0.95      0.91        41
       Both       0.98      0.93      0.95        95
    Neither       0.87      0.85      0.86        47
     Person       0.82      0.86      0.84        43

avg / total       0.91      0.90      0.90       226
```

# Future Work

I would like to create a new error metric to account for images from the 'Both' category being classified as 'Banana' or 'Person' to give the model partial credit.

With my current model, I would like to explore unfreezing more layers and gradually retraining to see if I can improve my model.

Next, I plan to add to my current structure to change it from a classification prediction to a regression prediction model. Then, I will train it on image data with corresponding dimensions of bananas and people in the image.

### Image Sources

Images are hyperlinked to their sources

