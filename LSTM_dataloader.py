import os 
import numpy as np
from sklearn.utils import shuffle 
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import sys 
import statistics 
from lstm_utils import *

dir_path = os.path.dirname(os.path.abspath(__file__))

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")


all_categories_string = ['Pop', 'Hip-Hop', 'Rock', 'Metal', 'Country', 'Jazz', 'Electronic', 'Folk', 'R&B', 'Indie']
all_categories = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
tag_to_ix = {0.0: 0, 1.0: 1, 2.0: 2, 3.0: 3, 4.0: 4, 5.0: 5,6.0: 6,7.0: 7, 8.0: 8, 9.0: 9}

#load data
params = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 6}
max_epochs = 30

dataset = song_dataset('./all_clean_data.csv')
trainset, valset = random_split(dataset, [20000,5000])

#train_loader = DataLoader(trainset,  batch_size = 2, shuffle = False)
#val_loader = DataLoader(valset,  batch_size = 2, shuffle = False)
train_loader = DataLoader(trainset,  **params)
val_loader = DataLoader(valset,  **params)

EMBEDDING_DIM = 32
HIDDEN_DIM = 32
#add one because of zero padding term
model = LSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, dataset.get_vocab_size(), dataset.get_label_size())
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.1)
#import pdb
#pdb.set_trace()
for epoch in range(max_epochs):  # again, normally you would NOT do 300 epochs, it is toy data
    train_correct = 0
    #training loop
#    for song, targets in train_loader:
    for i, batch in enumerate(train_loader):
#        print(batch[0].size())
        song, targets = batch
        targets = targets.squeeze(0)
#        import pdb; pdb.set_trace()
        model.zero_grad()

        #forward pass.
        tag_scores = model(song.squeeze(0).unsqueeze(-1))     

        #Compute the loss, gradients, and update the parameters by
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()
        
        #this works for no batching
        if categoryFromOutput(tag_scores[-1])[0] == targets[0].item():
            train_correct += 1
            
        if i > 999:
            break
    
#    calculate accuracy
    train_acc = 100*train_correct / i
        
    test_correct = 0
    #testing loop
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            song, targets = batch
            targets = targets.squeeze(0)
            model.zero_grad()
    
            #forward pass.
            tag_scores = model(song.squeeze(0).unsqueeze(-1))     
    
            #Compute the loss, gradients, and update the parameters by
            loss = loss_function(tag_scores, targets)
            
            #this works for no batching
            if categoryFromOutput(tag_scores[-1])[0] == targets[0].item():
                test_correct += 1
            
            if i > 99: 
                break
        
    #    calculate accuracy
        test_acc = 100*test_correct / i
        if np.mod(epoch, 1) == 0:
            print("epoch: ", epoch, "Training Acc. ", train_acc, " Testing Acc. ", test_acc )