import json
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')

# Load JSON data
with open("cs3430ai.json", "r") as file:
    data = json.load(file)

# Tokenization and preprocessing
def tokenize(sentence):
    return word_tokenize(sentence.lower())

def bag_of_words(tokenized_sentence, words):
    sentence_words = set(tokenized_sentence)
    bag = np.array([1 if w in sentence_words else 0 for w in words], dtype=np.float32)
    return bag

all_words = []
tags = []
xy = []

for intent in data["intents"]:
    tag = intent["intent"]
    tags.append(tag)
    for pattern in intent["patterns"]:
        w = tokenize(pattern)
        all_words.extend(w)
        xy.append((w, tag))

# Sort and remove duplicates
all_words = sorted(set(all_words))
tags = sorted(set(tags))
label_encoder = LabelEncoder()
tags_encoded = label_encoder.fit_transform(tags)

# Prepare training data
X_train = []
y_train = []
for (pattern_sentence, tag) in xy:
    bag = bag_of_words(pattern_sentence, all_words)
    X_train.append(bag)
    y_train.append(tags.index(tag))

X_train = np.array(X_train)
y_train = np.array(y_train)

# Define dataset class
class ChatDataset(Dataset):
    def __init__(self):
        self.n_samples = len(X_train)
        self.x_data = torch.tensor(X_train, dtype=torch.float32)
        self.y_data = torch.tensor(y_train, dtype=torch.long)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.n_samples

# Neural Network model
class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
        self.l1 = nn.Linear(input_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)
        out = self.relu(out)
        out = self.l3(out)
        return out

# Training parameters
batch_size = 8
hidden_size = 8
input_size = len(X_train[0])
output_size = len(tags)
learning_rate = 0.001
epochs = 1000

# Initialize dataset and dataloader
dataset = ChatDataset()
dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

# Model, loss, optimizer
model = NeuralNet(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Training loop
for epoch in range(epochs):
    for words, labels in dataloader:
        outputs = model(words)
        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}')

print("Training complete!")

# Save model data
torch.save({
    'model_state': model.state_dict(),
    'input_size': input_size,
    'hidden_size': hidden_size,
    'output_size': output_size,
    'all_words': all_words,
    'tags': tags
}, "chatbot_model.pth")

print("Model saved as chatbot_model.pth")
