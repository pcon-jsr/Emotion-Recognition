# Emotion-Recognition
In this project, I tried to predict the emotion of the person using deeplearning.
I trained [this](https://github.com/ritesh-nitjsr/Emotion-Recognition/blob/master/model.png) model, for around 50 epochs to achieve an accuracy of about 57% on the test set. I used the dataset from [this](https://www.kaggle.com/c/challenges-in-representation-learning-facial-expression-recognition-challenge) kaggle competetion to train my model. The data consists of 35587 48x48 pixel grayscale images of faces, expressing 7 emotions (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral). However,since examples for "Disgust" expression is too low, I merged it with "Anger", because both of them expresses almost same emotion.So now the images are just categorized into 6 categories (0=Angry/Disgust, 1=Fear, 2=Happy, 3=Sad, 4=Surprise, 5=Neutral).  

## Requirements:
   - Keras(Tensorflow Backend)
   - Pillow
   - OpenCV(2.3)
   - Numpy
   - Pandas
   - tqdm
   
