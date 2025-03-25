import torch
import torch.nn as nn
import torch.optim as optim

class RNNLM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(RNNLM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
    
    def forward(self, x, h_prev):
        embeds = self.embedding(x)
        rnn_out, h_next = self.rnn(embeds, h_prev)
        output = self.fc(rnn_out)
        return output, h_next

# Define parameters
VOCAB_SIZE = 10000
EMBEDDING_DIM = 128
HIDDEN_DIM = 512
SEQ_LENGTH = 5
BATCH_SIZE = 32
EPOCHS = 5

# Initialize model, optimizer, and loss function
model = RNNLM(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Generate dummy data
inputs = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
targets = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LENGTH))
h_prev = torch.zeros(1, BATCH_SIZE, HIDDEN_DIM)

# Training loop
for epoch in range(EPOCHS):
    optimizer.zero_grad()
    output, h_next = model(inputs, h_prev)
    loss = criterion(output.view(-1, VOCAB_SIZE), targets.view(-1))
    loss.backward()
    optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item()}")
