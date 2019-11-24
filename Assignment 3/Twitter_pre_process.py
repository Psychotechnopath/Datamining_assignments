import pandas as pd

twitter = pd.read_csv("C:/Users/Yme/Documents/MEGA/Master DSE/Data Engineering/Assignment/data/Sentiment Analysis Dataset 2.csv", skiprows=[8835,535881] , usecols = ['Sentiment' , 'SentimentText'])
twitter = twitter.rename(columns = {'Label': 'sentiment' , 'Text':'reviews'})

print('Shape of Dataset -> ' , twitter.shape)
