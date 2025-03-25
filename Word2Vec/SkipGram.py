import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np

class HierarchicalSoftmax(nn.Module):
    def __init__(self, vocab_size, hidden_dim, huffman_tree):
        super(HierarchicalSoftmax, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.huffman_tree = huffman_tree  # Precomputed Huffman tree
        self.out_layers = nn.ModuleDict()
        
        for word, path in huffman_tree.items():
            self.out_layers[word] = nn.Linear(hidden_dim, len(path), bias=False)

    def forward(self, hidden, target_words):
        batch_loss = 0
        for i, word in enumerate(target_words):
            if word in self.out_layers:
                logits = self.out_layers[word](hidden[i])
                targets = torch.tensor(self.huffman_tree[word], dtype=torch.float32, device=hidden.device)
                loss = F.binary_cross_entropy_with_logits(logits, targets)
                batch_loss += loss
        return batch_loss / len(target_words)

class SkipGram(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, huffman_tree):
        super(SkipGram, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.hidden_layer = nn.Linear(embedding_dim, hidden_dim)  # Project embeddings to hidden_dim
        self.h_softmax = HierarchicalSoftmax(vocab_size, hidden_dim, huffman_tree)

    def forward(self, center_word, target_words):
        embeds = self.embedding(center_word)  # (batch_size, embedding_dim)
        proj = F.relu(self.hidden_layer(embeds))  # Apply non-linearity
        loss = self.h_softmax(proj, target_words)  # Predict multiple target words
        return loss

# Example Huffman Tree
huffman_tree = {
    0: [0, 1],
    1: [1, 0],
    2: [0, 0, 1],
    3: [1, 1, 0]
}  # This should be generated based on word frequency

# Model Hyperparameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
CONTEXT_SIZE = 4  # Each center word predicts 4 context words

# Model Initialization
model = SkipGram(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, huffman_tree)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Dummy Training Example
center_words = torch.randint(0, VOCAB_SIZE, (32,))  # Each batch has one center word
context_words = torch.randint(0, VOCAB_SIZE, (32, CONTEXT_SIZE))  # Multiple context words per center

optimizer.zero_grad()
loss = model(center_words, context_words)
loss.backward()
optimizer.step()

print(f"Training Loss: {loss.item()}")
