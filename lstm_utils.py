import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils import data
from sklearn.utils import shuffle 

torch.manual_seed(1)

all_categories_string = ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie']
all_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

class LSTMTagger(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(LSTMTagger, self).__init__()
        self.hidden_dim = hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(embedding_dim, hidden_dim)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class song_dataset(data.Dataset):
    def __init__(self, csv_file):
        df = pd.read_csv(csv_file)
        self.songs = [] #set of songs
        self.genre = [] #set of labels
        self.vocab_size = 0
        self.labels = [] #all the unique labels in the dataset
        to_ix = {}        
        #load csv as dataframe
        for row in df.index:
#            lyrics = [w for w in df['lyrics'][row].split()]
            lyrics = []
            for w in df['lyrics'][row].split(): 
                if w not in to_ix:
                    to_ix[w] = len(to_ix)
                lyrics.append(to_ix[w])
            genre = df['genre'][row]
            if genre not in self.labels:
                self.labels.append(genre)
            
            tags = [genre for _ in range(len(lyrics))]
            self.songs.append(lyrics)
            self.genre.append(tags)
        
        self.vocab_size = len(to_ix)
        
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_label_size(self):
        return len(self.labels)
        
        
    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.songs[idx]
        y = self.genre[idx]
        
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
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

def get_data(csv_file, size = 1000, shuff = True):
    df = pd.read_csv(csv_file)
    if shuff:
        df = shuffle(df, random_state=2)
#        df = shuffle(df)
    df = df.iloc[:size]
    data = []
    for row in df.index:
        lyrics = df['lyrics'][row].split()
        target = df['genre'][row]
        tags = [target for _ in range(len(lyrics))]
        data.append((lyrics, tags))
    
    return data

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)

#convert sentences of words to sequences of numbers, each number corresponding to a specific word
def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


def categoryFromOutput(output): 
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() 
    return all_categories[category_i], all_categories_string[category_i]