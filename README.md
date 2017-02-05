# question_classification

Note: This code was written in Python-2.7 Anaconda Environment.So i used Scikit learn (sklearn) package for the modelling of the data,Numpy for converting lists into arrays and punctuation from string package.  

To run this code following  packages are required:
scikit-learn 0.18.1
numpy 1.11.3
string

# Pre-processing:
As we see the train data contains question marks,punctuations,articles,"'s" before what
In this process,the text data will be removed with all these things and sent for vector transformation

Also,our labelled train data with character labels are converted to numerical labels for modelling with the below map
{"what" :1,"when":2,"who":3,"unknown":4,"affirmation":5}

# Vectorization:
For the vectorization of text data, we used word count vectorizer to convert a text query into vector.
It will take the whole text corpus and build a vocabulary which consists of set of possible word in the text corpus
So our each train data will convert into a vector of 1's and 0's in which 1 represents the frequency of the data

TF-IDF vectorization can also be performed but in this case Countvectorizer seems better in performance

# Learning 

For the best modelling of text data,a multinominal Naives bayes classifier is used with the default parameters
We performed Also performed k-flod cross validation to check our model

In our model we observed that the cross validation mean score to be around 0.89 and it becomes 0.91 when we used bi-grams too

# Prediction 
For the prediction,we should convert our test data into vectors and sent for the model to predict.
So, the output contains the labels attached to them

# Results and Accuracy

We tested the data on file "testData.txt" which is same as train data and that gives below performance results
Accuracy : 0.978000
Variance score: 0.98

This accuracy in turn will be good if we took bi-grams also in conideration
