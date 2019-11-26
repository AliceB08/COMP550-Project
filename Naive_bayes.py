import numpy as np
import os
import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
import string
import nltk

def genre_to_int(genre):
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
    if genre == 'Not Available': return None 
    if genre == 'Other': return None
    
#pass in lyrics
def pre_process(songs):
    all_songs = []
    all_labels = []
    for i,song in zip(songs.index,songs['lyrics']):
#        print(i)
        if songs['genre'][i] == 'Not Available' or songs['genre'][i] == 'Other':
            continue
        else:
            #strip punctuation
            songs['lyrics'][i] = ''.join(ch for ch in songs['lyrics'][i] if ch not in exclude)
        if len(np.unique(songs['lyrics'][i].split(' '))) < 5:
            continue
        elif detect(song) == 'en':
            chancon = song.replace('\n', ' ').replace('(','').replace(')','').split(' ')
            new_chancon = []
            for ww in chancon:
                if len(ww) == 0:
                    chancon.remove(ww)
                    continue
                elif ww[0] == '[' or ww[-1] == ']':
                    chancon.remove(ww)
#                elif ww[-1] in set(string.punctuation):
#                    ww = ww[:-1]
                new_chancon.append(ww)
            songs['lyrics'][i] = ' '.join(new_chancon)
            songs['genre'][i] = genre_to_int(songs['genre'][i])
            all_labels.append(songs['genre'][i])
            all_songs.append(' '.join(new_chancon))
    return all_songs,all_labels

def find_features(document,word_features):
    words = set(document.split(' '))
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

corpus_root = os.path.dirname(os.path.realpath(__file__)).split('/')
data_root = '/'.join(corpus_root[:10]) + '/data/kaggle'
exclude = set(string.punctuation)
exclude.add('\n')

N = 5000
songs = pd.read_csv('./lyrics.csv')
songs = songs[pd.notnull(songs['lyrics'])]
songs = songs.drop(songs.index[N:])
songs = songs.sample(frac=1)
song_list,label_list = pre_process(songs)
for label in label_list:
    label = genre_to_int(label)

#generate bag of words features
all_words = [];
stop = stopwords.words('english')
for doc in song_list:
    for w in doc.split(' '):
        if w.lower() not in stop:
            all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
word_features = [w for w in list(all_words.most_common())[:3000]]
features = [(find_features(song,word_features), genre) for (song,genre) in zip(song_list,label_list)]
training_set = features[:int(0.8*len(features))]
testing_set = features[int(0.8*len(features))+1:]

clf = nltk.NaiveBayesClassifier.train(training_set)

acc = (nltk.classify.accuracy(clf,testing_set))*100
print(" acc.", acc)