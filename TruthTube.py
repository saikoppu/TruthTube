from selenium import webdriver
from webdriver_manager.firefox import GeckoDriverManager
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import time
import os
import nltk
import re
import string
import seaborn as sns

from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer, LancasterStemmer
from nltk.stem.snowball import SnowballStemmer
from nltk.corpus import stopwords
from nltk.corpus import wordnet
from string import punctuation

def ScrapComment(url):
    option = webdriver.FirefoxOptions()
    option.add_argument("--headless")
    driver = webdriver.Firefox(executable_path=GeckoDriverManager().install())
    driver.get(url)
    prev_h = 0
    x = time.time() + 100
    y = 0
    while x > y:
        height = driver.execute_script("""
                function getActualHeight() {
                    return Math.max(
                        Math.max(document.body.scrollHeight, document.documentElement.scrollHeight),
                        Math.max(document.body.offsetHeight, document.documentElement.offsetHeight),
                        Math.max(document.body.clientHeight, document.documentElement.clientHeight)
                    );
                }
                return getActualHeight();
            """)
        driver.execute_script(f"window.scrollTo({prev_h},{prev_h + 200})")
        # ADAPT THIS SLEEP BASED ON NETWORK
        time.sleep(0.3)
        prev_h += 200 
        y = time.time()
        if prev_h >= height:
            break
    soup = BeautifulSoup(driver.page_source, 'html.parser')
    driver.quit()
    title_text_div = soup.select_one('#container h1')
    title = title_text_div and title_text_div.text
    comment_div = soup.select("#content #content-text")
    comment_list = [x.text for x in comment_div]
    comment_list.append(title)
    return comment_list


urls = [
        "https://www.youtube.com/watch?v=hSlb1ezRqfA",
        "https://www.youtube.com/watch?v=3DJlwNCGFIY",
        "https://www.youtube.com/watch?v=cgNQgcUgq0U",
        "https://www.youtube.com/watch?v=MkE_EwO76b0",
        'https://www.youtube.com/watch?v=XVv6mJpFOb0',
    ]
listComments = ScrapComment(urls[0])

REPLACE_NO_SPACE = re.compile("[.;:!\'?,\"()\[\]]")
REPLACE_WITH_SPACE = re.compile("(<br\s*/><br\s*/>)|(\-)|(\/)")

def preprocess_reviews(reviews):
    reviews = [REPLACE_NO_SPACE.sub("", line.lower()) for line in reviews]
    reviews = [REPLACE_WITH_SPACE.sub(" ", line) for line in reviews]
    return reviews

cleanedComments = preprocess_reviews(listComments)
dataSet = []
dataSet.append(cleanedComments)

nltk.download('vader_lexicon')
sentiments = SentimentIntensityAnalyzer()
posList = []
negList = []
neutralList = [] 
score = []
for i in range(0, len(dataSet[0])):
    pos = sentiments.polarity_scores(dataSet[0][i])["pos"]
    neg = sentiments.polarity_scores(dataSet[0][i])["neg"]
    neu = sentiments.polarity_scores(dataSet[0][i])["neu"]
    comp = sentiments.polarity_scores(dataSet[0][i])["compound"]
    posList.append(pos)
    negList.append(neg)
    neutralList.append(neu)
    score.append(comp)
sentiment = []
sentList = [0, 0, 0]
for i in score:
    if i >= 0.1 :
        sentiment.append('Positive')
        sentList[0] += 1
    elif i <= -0.1 :
        sentiment.append('Negative')
        sentList[2] += 1
    else:
        sentiment.append('Neutral')
        sentList[1] += 1
finalScore = 0
for i in sentiment:
    if i == "Positive":
        finalScore += 1
    elif i == "Negative":
        finalScore -= 1

score = [i for i in score if i != 0.0]
print(score)
print(cleanedComments)
print(finalScore)
print(str(sum(score)/len(score)))

if finalScore > 0:
    print("Most people like this video :)")
elif finalScore < 0:
    print("Most people dislike this video :(")
else:
    print("Neutral video -- not much to say :|")

words = ["Positive", "Neutral", "Negative"]
graph = sns.barplot(words, sentList)
graph.set(title = "YouTube Comment's Sentiment Analyzer", xlabel = "Type of Comment", ylabel = "Number of Comments")
plt.savefig("bar.png")
plt.show()