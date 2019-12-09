import os 
import numpy as np
from sklearn.utils import shuffle 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys 
import statistics 
from lstm_utils import *

dir_path = os.path.dirname(os.path.abspath(__file__))


all_categories_string = ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie']
all_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tag_to_ix = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5,6.0: 6,7.0: 7, 8.0: 8, 9.0: 9}

#load data
training_data = get_data('./train_data.csv', size = 1000)
testing_data = get_data('./test_data.csv', size = 200)

#convert words to a number
word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
for sent, tags in testing_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

# These will usually be more like 32 or 64 dimensional.
# We will keep them small, so we can see how the weights change as we train.
EMBEDDING_DIM = 32
HIDDEN_DIM = 32

max_epochs = 30

model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)

for epoch in range(max_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    train_correct = 0
    
    #training loop
    for song, tags in training_data:
      
        model.zero_grad()
        sentence_in = prepare_sequence(song, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        #forward pass.
        tag_scores = model(sentence_in)
        
        #Compute the loss, gradients, and update the parameters by
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
        #this works for no batching
        if categoryFromOutput(tag_scores[-1])[0] == targets[0].item():
            train_correct += 1
    
    #calculate accuracy
    train_acc = 100*train_correct / len(training_data)
        
    test_correct = 0
    #testing loop
    with torch.no_grad():
         #training loop
        for sentence, tags in testing_data:
            sentence_in = prepare_sequence(sentence, word_to_ix)
            targets = prepare_sequence(tags, tag_to_ix)
            tag_scores = model(sentence_in)
            
            loss = loss_function(tag_scores, targets)
            
            #this works for no batching
            if categoryFromOutput(tag_scores[-1])[0] == targets[0].item():
                test_correct += 1
                
        test_acc = 100*test_correct / len(testing_data)
        if np.mod(epoch, 1) == 0:
            print("epoch: ", epoch, "Training Acc. ", train_acc, " Testing Acc. ", test_acc )