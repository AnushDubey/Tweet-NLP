# Introduction
This assignment involves generating new tweets based on the user's existing tweets. The goal is to use a deep learning model to generate new text that is similar in style and content to the original tweets. The model used in this assignment is a type of Recurrent Neural Network (RNN) called Long Short-Term Memory (LSTM) model. The dataset used for this assignment is the Kaggle Sentiment140 dataset which contains 1.6 million tweets with their corresponding sentiments. Link for dataset- https://www.kaggle.com/datasets/kazanova/sentiment140?resource=download

# Approach
I have considered one user for generation since the dataset was very large.

Data Preprocessing: The first step in this assignment is to preprocess the dataset. This involves cleaning the data, tokenizing the text, and converting the text into sequences of integers.

Model Architecture: The next step is to build the LSTM model. This model takes in a sequence of integers and predicts the next word in the sequence. The model is trained on the preprocessed data and the weights of the trained model are saved.

Generating New Tweets: To generate new tweets, the saved weights of the trained LSTM model are loaded, and a seed text is given as input to the model. The model then predicts the next word in the sequence and adds it to the generated tweet. This process is repeated until the desired length of the tweet is reached.

Adjusting the randomness of the generated tweets: The randomness of the generated tweets can be adjusted using the temperature parameter. A higher temperature value results in more randomness in the generated text, while a lower temperature value results in more conservative text.

# Files

tweetNLP.ipynb: This is the Jupyter notebook containing the code for generating new tweets.

model.h5: This is the saved weights of the trained LSTM model.

# Dependencies
tensorflow
keras
pandas
numpy
nltk
# How to Use
 * Install Jupyter Notebook on your computer
 * Go to your desired folder and clone the repository using the following command:
```
  $ git clone https://github.com/AnushDubey/Tweet-NLP.git
```
 * Change the current directory using following command:
```
  $ cd Tweet-NLP
```
 * Run cmd in the folder
 * In the Command Prompt window type ```jupyter notebook```

# Conclusion
In this assignment, we used a Long Short-Term Memory (LSTM) model to generate new tweets based on the user's existing tweets. The model was trained on the Kaggle Sentiment140 dataset and the weights of the trained model were saved. We were able to generate new tweets by providing a seed text to the model and adjusting the randomness of the generated text using the temperature parameter.
