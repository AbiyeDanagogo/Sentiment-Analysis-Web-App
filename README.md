# Sentiment-Analysis-Web-App

This  project is for a sentiment analysis web app that predicts the positive sentiment of a piece of text.

The machine learning model used to make predictions was the multinomial Naive Bayes classifier from scikit learn. The model was trained on the [Twitter Sentiment Analysis Training Corpus dataset.](http://thinknook.com/twitter-sentiment-analysis-training-corpus-dataset-2012-09-22/)
It contains 1,578,627 classified tweets, each row is marked as 1 for positive sentiment and 0 for negative sentiment.

The model was saved using dill. Dill is a python module which can be used to store python objects to a file, and also send the objects across a network as a byte stream.

The web app was created using  Flask. Flask is a microframework for python that allows users to build websites and web apps easily. The web app was deployed onto the internet using Heroku. The url of the web app is [https://sentiment-analyser-abiye.herokuapp.com/index](https://sentiment-analyser-abiye.herokuapp.com/index)
