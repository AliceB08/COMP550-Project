import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import Normalizer, FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt 
from PIL import Image 
from wordcloud import WordCloud, STOPWORDS
import numpy as np 
from nltk.corpus import stopwords
from collections import Counter 
 
#*************** Initialize *******************

swords = set(stopwords.words('english')) 

#*********** Customer Preprocessing ********
def genre_to_int(data):  
    genre = data['genre']
    if genre == 'Pop': return 0
    if genre == 'Hip-Hop': return 1
    if genre == 'Rock': return 2
    if genre == 'Metal': return 3
    if genre == 'Country': return 4
    if genre == 'Jazz': return 5
    if genre == 'Electronic': return 6
    if genre == 'Folk': return 7
    if genre == 'R&B': return 8
    if genre == 'Indie': return 9
    if genre == 'Not Available': return 10 #None
    if genre == 'Other': return 10 #None
    
def clean(data): 
    row = data['lyrics']
    row = str(row).lower().replace("\n", ' ')
    return row 

#*************** Read Data *******************
csv = pd.read_csv('./lyrics.csv')
csv = csv.iloc[:20000, :]
csv['genre'] = csv.apply(genre_to_int, axis=1)
csv['lyrics'] = csv.apply(clean, axis=1)
#print("number of rows", len(csv))
#csv.dropna(axis = 0, how="any", inplace=True) #drop row if any null in it 
#print("number of rows after dropping null, len(csv))
#labels = csv['genre']  
#csv['lyrics'] = csv.apply(clean, axis=1)
#lyrics = csv['lyrics']
#hh_lyrics = csv.loc[csv['genre'] == 1]['lyrics']
#print(len(rb_lyrics)) 


def dist(labels): 
    sns.distplot(labels, kde=False, rug=True)
    # ax.to_file('label_distribution.png')
    plt.show() 
def corr_plot(csv):  
    sns.heatmap(csv.corr(method='spearman'), annot=True)
    plt.show()
def cloud(lyrics): 
    np_lyrics = lyrics.to_numpy(dtype='str')
    full_text = ""
    for row in range(len(np_lyrics)):
        l = np_lyrics[row].replace('\n', ' ')
        full_text += l 
    print('done loading')   
    maskArray = np.array(Image.open('cloud.png'))
    cloud = WordCloud(background_color = "white", max_words = 200, mask=maskArray, stopwords = swords)
    cloud.generate(full_text)
    cloud.to_file("./Clouds/hhWordCloud2.png")
def popularity_of_words(lyrics): 
    np_lyrics = lyrics.to_numpy(dtype='str')
    word_tokens = {}
    for row in range(len(np_lyrics)):
        l = np_lyrics[row].replace('\n', ' ')
        for w in l.split():
            if w in word_tokens and w not in swords:  
                word_tokens[w] += 1
            if w not in word_tokens and w not in swords:   
                word_tokens[w] = 1 
            else: continue 
        print("reading", row, "of", len(np_lyrics))
    word_freq = sorted(list(zip(word_tokens.values(), word_tokens.keys())), reverse=True)
    print(word_freq[:25])

#corr_plot(hh_lyrics)
dist(csv['genre'])    
#cloud(hh_lyrics)
#popularity_of_words(lyrics)