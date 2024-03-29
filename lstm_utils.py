import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
import pandas as pd
from torch.utils.data import DataLoader, random_split
from torch.utils import data
from sklearn.utils import shuffle 
from sklearn.model_selection import train_test_split



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
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first = True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)

    def forward(self, sentence):
#        sentence = sentence.unsqueeze(-1)
        embeds = self.word_embeddings(sentence)
        lstm_out, _ = self.lstm(embeds.view(len(sentence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space, dim=1)
        return tag_scores

class song_dataset(data.Dataset):
    def __init__(self, csv_file):
        self.songs = [] #set of songs
        self.genre = [] #set of labels
        self.vocab_size = 0
        self.labels = [] #all the unique labels in the dataset
        self.song_size = 0
        
        df = pd.read_csv(csv_file)
        to_ix = {}        
        #load csv as dataframe and process data
        for row in df.index:
#            lyrics = [w for w in df['lyrics'][row].split()]
            lyrics = []
            if len(df['lyrics'][row].split()) > self.song_size:
                self.song_size = len(df['lyrics'][row].split())
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
        
        to_ix['zero_pad_token'] = len(to_ix)
        
        #zero pad songs
        for i in range(len(self.songs)):
            if len(self.songs[i]) == self.song_size:
                continue
            else:
                self.songs[i] = self.songs[i] +[to_ix['zero_pad_token']]*(self.song_size - len(self.songs[i]))
                self.genre[i] = self.genre[i] + [self.genre[i][-1]]*(self.song_size - len(self.genre[i]))                

        self.vocab_size = len(to_ix)
        self.words = to_ix
        
    def get_vocab_size(self):
        return self.vocab_size
    
    def get_label_size(self):
        return len(self.labels)
    
    def get_song_size(self):
        return self.song_size
    
    def get_vocabulary(self):
        return self.words
        
    def __len__(self):
        return len(self.songs)

    def __getitem__(self, idx):
        'Generates one sample of data'
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        X = self.songs[idx]
        y = self.genre[idx]
        
        return torch.tensor(X, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    
# def genre_to_int(row):
#     genre = row['genre']
#     if genre == 'Pop': return 0
#     if genre == 'Hip-Hop': return 1
#     if genre == 'Rock': return 2
#     if genre == 'Metal': return 3
#     if genre == 'Country': return 4
#     if genre == 'Jazz': return 5
#     if genre == 'Electronic': return 6
#     if genre == 'Folk': return 7
#     if genre == 'R&B': return 8
#     if genre == 'Indie': return 9
#     if genre == 'Not Available': return None 
#     if genre == 'Other': return None

# def get_data(csv_file, size = 1000, shuff = True):
#     df = pd.read_csv(csv_file)
#     df['genre'] = df.apply(genre_to_int, axis=1)
#     df.dropna(axis = 0, how="any", inplace=True)
    
#     if shuff:
#         df = shuffle(df, random_state=2)
#     df = df.iloc[:size]
#     data = []
#     for row in df.index:
#         try: 
#             lyrics = df['lyrics'][row].split()
#             target = df['genre'][row]
#             tags = [target for _ in range(len(lyrics))]
#             data.append((lyrics, tags))
#         except: print()
    
#     return data

def balance(df): 
    
    lengths = [] 

    pop = df.loc[df['genre'] == 0]
    rock = df.loc[df['genre'] == 1]
    hh = df.loc[df['genre'] == 2]
    metal = df.loc[df['genre'] == 3]
    country = df.loc[df['genre'] == 4]
    elec = df.loc[df['genre'] == 5]
    folk = df.loc[df['genre'] == 6]
    rb = df.loc[df['genre'] == 7]
    indie = df.loc[df['genre'] == 8]

    # df2 = pd.DataFrame()
    all_classes = [i for i in range(9)]
    for c in all_classes: 
        subset = df.loc[df['genre'] == c]
        lengths.append(len(subset))
    print(all_classes)
    print(lengths)
    # bNum = min(lengths)
    bNum = 1000
    pop = shuffle(pop, random_state = 2)
    rock = shuffle(rock, random_state = 2)
    hh = shuffle(hh, random_state = 2)
    metal = shuffle(metal, random_state= 2)
    country = shuffle(country, random_state = 2)
    elec = shuffle(elec, random_state = 2)
    folk = shuffle(folk, random_state= 2)
    rb = shuffle(rb, random_state = 2)
    indie = shuffle(indie, random_state = 2)

    pop = pop.iloc[:bNum]
    rock = rock.iloc[:bNum]
    hh = hh.iloc[:bNum]
    metal = metal.iloc[:bNum]
    country = country.iloc[:bNum]
    elec = elec.iloc[:bNum]
    folk = folk.iloc[:bNum]
    rb = rb.iloc[:bNum]
    indie = indie.iloc[:bNum]

    df2 = pd.concat([pop, rock, hh, metal, country, elec, folk, rb, indie], axis=0)
    lengths = []
    for c in all_classes: 
        subset = df2.loc[df['genre'] == c]['lyrics']
        lengths.append(len(subset))
    print(lengths)
    print('************BALANCED****************')
    print(df2)
    print('************BALANCED****************')
    return df2

def genre_to_int(row):
    genre = row['genre']
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


def get_data(csv_file, size, shuff, mode):
    df = pd.read_csv(csv_file)
    print(df)
    # df['genre'] = df.apply(genre_to_int, axis=1)
    df.dropna(axis = 0, how="any", inplace=True)
    df = balance(df)
    # if shuff:
    #     df = shuffle(df, random_state=2)
    # df = df.iloc[:size]

    train_data = []
    test_data = [] 

    all_X = df['lyrics']
    all_Y = df['genre']
    X_train, X_test, y_train, y_test = train_test_split(all_X, all_Y, test_size = .2, random_state = 3, stratify=all_Y, shuffle=True)
    
    if mode == 'train': 
        train_df = pd.concat([X_train,y_train], axis=1)
        print(train_df)
        for row in train_df.index:
            try: 
                lyrics = df['lyrics'][row].split()
                target = df['genre'][row]
                tags = [target for _ in range(len(lyrics))]
                train_data.append((lyrics, tags))
            except: print()
        return train_data
    
    if mode == 'test':
        test_df = pd.concat([X_test,y_test], axis=1)
        
        for row in test_df.index:
            try: 
                lyrics = df['lyrics'][row].split()
                target = df['genre'][row]
                tags = [target for _ in range(len(lyrics))]
                test_data.append((lyrics, tags))
            except: print()
    
        return test_data

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