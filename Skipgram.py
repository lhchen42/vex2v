import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

class Model(nn.Module):
    def __init__(self, vocabulary_size, embedding_size):
        super(Model, self).__init__()
        self.vocabulary_size = vocabulary_size
        self.embedding_size = embedding_size
        self.embeddings = nn.Embedding(self.vocabulary_size, self.embedding_size)
        self.linear1 = nn.Linear(self.embedding_size, self.vocabulary_size)

    
    def forward(self, input):
        embeds = self.embeddings(input)
        #print(embeds.shape)
        out = self.linear1(F.relu(embeds))
        log_probs = F.log_softmax(out, dim=1)
        
        return log_probs
    
    def get_embedding(self, input):
        out_vector = self.embeddings(input)
        return out_vector 
        