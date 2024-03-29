from __future__ import unicode_literals, print_function, division
from io import open 
import glob 
import os 
import pandas as pd 
from nltk.corpus import stopwords, words 
import re 
from nltk import tokenize
import numpy as np 
from sklearn.utils import shuffle 
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score, accuracy_score, auc, f1_score, recall_score

import nltk
nltk.download('words')

os.environ['KMP_DUPLICATE_LIB_OK']='True'



def balance(df): 
    
    lengths = [] 

    pop = df.loc[df['genre'] == 'Pop']
    rock = df.loc[df['genre'] == 'Rock']
    hh = df.loc[df['genre'] == 'Hip-Hop']
    metal = df.loc[df['genre'] == 'Metal']
    country = df.loc[df['genre'] == 'Country']
    elec = df.loc[df['genre'] == 'Electronic']
    folk = df.loc[df['genre'] == 'Folk']
    rb = df.loc[df['genre'] == 'R&B']
    indie = df.loc[df['genre'] == 'Indie']

    genre_list = ['Pop', 'Rock', 'Hip-Hop', 'Metal', 'Country', 'Electronic', 'Folk', 'R&B', 'Indie']
    print(len(pop))

    for c in genre_list: 
        subset = df.loc[df['genre'] == c]
        lengths.append(len(subset))
    
    print(lengths)
    bNum = min(lengths)
    # bNum = 1000
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
    for c in genre_list: 
        subset = df2.loc[df['genre'] == c]['lyrics']
        lengths.append(len(subset))
    print(lengths)
  
    return df2

def remove_unknown(row):
    genre = row['genre']
    
    if genre == 'Not Available': return None 
    if genre == 'Other': return None


#LOAD DATA

df = pd.read_csv('./lyrics.csv')
df = df[df.lyrics != 'Instrumental']
# df['genre'] = df.apply(remove_unknown, axis=1)
df.dropna(axis = 0, how="any", inplace=True)
#df = balance(df)


REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))
ENGLISH = set(nltk.corpus.words.words())


def write_to_txt(df, name): 
    lyrics = df.loc[df['genre'] == name]['lyrics']
    np_lyrics = lyrics.to_numpy(dtype='str')
    full_text = ""
    with open('./lyrics/'+name+'_lyrics.txt', "w+") as f: 
        for row in range(len(np_lyrics)):
            try: 
                l = np_lyrics[row].replace('\n', ' ').lower()
                l = REPLACE_BY_SPACE_RE.sub(' ', l) 
                l = BAD_SYMBOLS_RE.sub('', l)
                l = ' '.join(word for word in l.split() if word not in STOPWORDS)
                f.write(l + '\n')
            except: print(l)
        print('done loading', name)   
genre_list = ['Pop', 'Rock', 'Hip-Hop', 'Metal', 'Country', 'Electronic', 'Folk', 'R&B', 'Indie']

for g in genre_list: 
    if not os.path.exists('./lyrics/'+g+"_lyrics.txt"): 
        write_to_txt(df, g)

def findFiles(path): return glob.glob(path)

print(findFiles('lyrics/*.txt'))

import unicodedata
import string 

all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

#turn unicode string to plain ASCII 
def unicodeToAscii(s): 
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )
#print(unicodeToAscii('Ślusàrski'))

#Build category liens dictionary, a list of names per language 
category_lines = {} 
all_categories = [] 

#Read a file and split into lines 
def readLines(filename): 
    lines = open(filename, encoding="utf-8").read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

for filename in findFiles('lyrics/*.txt'): 
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines
n_categories = len(all_categories)

print("EVALUTATING LYRICS FOR", all_categories)

#CONVERT DATA TO TENSOR
import torch 
def letterToIndex(letter): 
    return all_letters.find(letter)

def lineToTensor(line): #one hot encoding
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line): 
        tensor[li][0][letterToIndex(letter)] = 1 
    return tensor 

print(lineToTensor(category_lines['Hip-Hop_lyrics'][:1]).size())

#BUILD NETWORK 
import torch.nn as nn 

class RNN(nn.Module): 
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()

        self.hidden_size = hidden_size

        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)

n_hidden = 128
rnn = RNN(n_letters, n_hidden, n_categories)

print("net param", rnn)

#TESTING RNN
# input = lineToTensor(category_lines['hh_lyrics'][:1])
# hidden = torch.zeros(1, n_hidden)
# output, next_hidden = rnn(input[0], hidden)
# print(output)

#TRAINING RNN
def categoryFromOutput(output): 
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() 
    return all_categories[category_i], category_i

# print(categoryFromOutput(output))

import random 

def randomChoice(l): 
    return l[random.randint(0, len(l) - 1)]

def randomTrainingExample(): 
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    category_tensor = torch.tensor([all_categories.index(category)], dtype=torch.long)
    line_tensor = lineToTensor(line)
    return category, line, category_tensor, line_tensor
criterion = nn.NLLLoss()

#Each loop of training will: 
    #Create input and target tensors 
    #Create zeroed initial hidden state 
    #Read each letter in and keep hidden state for next letter 
    #Compare final output to target 
    #Back propagate 
    #Return output and loss 

learning_rate = 0.01

def train(category_tensor, line_tensor): 
    hidden = rnn.initHidden()
    rnn.zero_grad() 
    for i in range(line_tensor.size()[0]): 
        
        output, hidden = rnn(line_tensor[i], hidden)

    
    loss = criterion(output, category_tensor)
    loss.backward() 
    
    for p in rnn.parameters(): 
        p.data.add_(-learning_rate, p.grad.data)

    return output,loss.item() 

import time 
import math 

n_iters = 1000 #100000
print_every = 50 #5000
plot_every = 10 #1000 

current_loss = 0 
all_losses = [] 

def timeSince(since): 
    now = time.time() 
    s = now - since 
    m = math.floor(s/60)
    s -= m * 60 
    return '%dm %ds' % (m, s)
start = time.time() 
print('STARTING TRAINING')
y_pred, y_true = [] , [] 
for iter in range(1, n_iters+1): 
    category, line, category_tensor, line_tensor = randomTrainingExample()
    try: 
        output, loss = train(category_tensor, line_tensor)
        current_loss += loss 

        guess, guess_i = categoryFromOutput(output)
        y_pred.append(guess)
        y_true.append(category)
        if iter % print_every == 0:
            guess, guess_i = categoryFromOutput(output)
            correct = '✓' if guess == category else '✗ (%s)' % category
            print('%d %d%% (%s) %.4f %s / %s %s' % (iter, iter / n_iters * 100, timeSince(start), loss, line[:20], guess, correct))
            print('accuracy', accuracy_score(y_true, y_pred))
        # Add current loss avg to list of losses
        if iter % plot_every == 0:
            all_losses.append(current_loss / plot_every)
            current_loss = 0
    except: print(category, line)

#PLOT TRAINING LOSS 
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure()
plt.plot(all_losses)
plt.show()

print("EVALUATING PERFORMANCE")

#EVALUATE PERFORMANCE 
confusion = torch.zeros(n_categories, n_categories)
n_confusion = 1000 #10000

def evaluate(line_tensor): 
    hidden = rnn.initHidden()
    for i in range(line_tensor.size()[0]): 
        output, hidden = rnn(line_tensor[i], hidden)
    return output 

for i in range(n_confusion): 
    category, line, category_tensor, line_tensor = randomTrainingExample()
    output = evaluate(line_tensor)
    guess, guess_i = categoryFromOutput(output)
    category_i = all_categories.index(category)
    confusion[category_i][guess_i] += 1

for i in range(n_categories): 
    confusion[i] = confusion[i]/ confusion[i].sum() 

# Set up plot
fig = plt.figure()
ax = fig.add_subplot(111)
cax = ax.matshow(confusion.numpy())
fig.colorbar(cax)

# Set up axes
ax.set_xticklabels([''] + all_categories, rotation=90)
ax.set_yticklabels([''] + all_categories)

# Force label at every tick
ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# sphinx_gallery_thumbnail_number = 2
plt.show()