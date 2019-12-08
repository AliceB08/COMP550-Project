import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd 
import os 
from sklearn.utils import shuffle 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
torch.manual_seed(1)
import sys 
import statistics 


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
    if genre == 'Not Available': return None 
    if genre == 'Other': return None

#LOAD DATA 
df = pd.read_csv('./lyrics.csv')
df['genre'] = df.apply(genre_to_int, axis=1)
df.dropna(axis = 0, how="any", inplace=True)
df = df[df.lyrics != 'Instrumental']
df = shuffle(df, random_state=2)
df = df.iloc[:2]
#print(df['genre'].unique())
sys.exit
all_categories_string = ['Pop', 'Rock', 'Hip-Hop', 'Metal', 'Country', 'Electronic', 'Folk', 'R&B', 'Indie']
all_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
training_data = []

for row in df.index:
    lyrics = df['lyrics'][row].split()
    #print(lyrics)
    #print(lyrics)
    target = df['genre'][row]
    tags = [target for t in range(len(lyrics))]
    training_data.append((lyrics, tags))



def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    return torch.tensor(idxs, dtype=torch.long)


# training_data = [
#     ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
#     ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
# ]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
#print(word_to_ix)
tag_to_ix = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5,6.0: 6,7.0: 7, 8.0: 8, 9.0: 9}

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


def categoryFromOutput(output): 
    top_n, top_i = output.topk(1)
    category_i = top_i[0].item() 
    return all_categories[category_i], all_categories_string[category_i]

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

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

# See what the scores are before training
# Note that element i,j of the output is the score for tag j for word i.
# Here we don't need to train, so the code is wrapped in torch.no_grad()
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)
#     print(tag_scores)

for epoch in range(10):  # again, normally you would NOT do 300 epochs, it is toy data
    for sentence, tags in training_data:
        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors of word indices.
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        # Step 3. Run our forward pass.
        tag_scores = model(sentence_in)
        print(epoch, sent[:5], categoryFromOutput(tag_scores[-1]))
        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

# See what the scores are after training
# with torch.no_grad():
#     inputs = prepare_sequence(training_data[0][0], word_to_ix)
#     tag_scores = model(inputs)

#     # The sentence is "the dog ate the apple".  i,j corresponds to score for tag j
#     # for word i. The predicted tag is the maximum scoring tag.
#     # Here, we can see the predicted sequence below is 0 1 2 0 1
#     # since 0 is index of the maximum value of row 1,
#     # 1 is the index of maximum value of row 2, etc.
#     # Which is DET NOUN VERB DET NOUN, the correct sequence!
#     print(len(tag_scores))
#     print(categoryFromOutput(tag_scores[-1])) 