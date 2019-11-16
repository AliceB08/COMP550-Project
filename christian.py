import numpy as np
import os
import pandas as pd
from langdetect import detect
from sklearn.feature_extraction.text import CountVectorizer

#pass in lyrics
def pre_process(songs):
    all_songs = []
    for song in songs['lyrics']:
        if detect(song) == 'en':
            chancon = song.replace('\n', ' ').replace('(','').replace(')','').split(' ')
            for w in chancon:
                if len(w) == 0:
                    chancon.remove(w)
                elif w[0] == '[' or w[-1] == ']':
                    chancon.remove(w)
            all_songs.append(' '.join(chancon))
    return all_songs
    

corpus_root = os.path.dirname(os.path.realpath(__file__)).split('/')
data_root = '/'.join(corpus_root[:10]) + '/data/kaggle'

songs = pd.read_csv(data_root + '/lyrics.csv')
songs = songs[pd.notnull(songs['lyrics'])]
songs = songs.drop(songs.index[100:])
song_list = pre_process(songs)

vectorizer = CountVectorizer(stop_words = 'english',max_features=5000)
X = vectorizer.fit_transform(song_list)