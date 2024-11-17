import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np


# Set the device
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print(f"Using device: {device}")
# Load the embeddings from the file pkl file
with open("data/embeddings.pkl", "rb") as f:
    embeddings_dict = pickle.load(f)

embeddings_loaded = np.array(list(embeddings_dict.values()))
embedding_size = embeddings_loaded.shape[1]

print(f"Embeddings shape: {embeddings_loaded.shape}")


# Define the PyTorch Dataset
class EmbeddingDataset(Dataset):
    def __init__(self, embeddings):
        self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        original_embedding = self.embeddings[idx]
        # Add Gaussian noise to create corrupted input
        noise = np.random.normal(0, 0.1, original_embedding.shape)
        corrupted_embedding = original_embedding + noise
        return torch.FloatTensor(corrupted_embedding), torch.FloatTensor(
            original_embedding
        )


# Create the dataset and dataloader
dataset = EmbeddingDataset(embeddings_loaded)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)


# Define the denoising autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self, input_size=3072):
        super(DenoisingAutoencoder, self).__init__()
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_size, 2048),
            nn.ReLU(True),
            nn.Linear(2048, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 512),
            nn.ReLU(True),
        )
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 2048),
            nn.ReLU(True),
            nn.Linear(2048, input_size),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


# Initialize the model, loss function, and optimizer
model = DenoisingAutoencoder(input_size=embedding_size).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

# Training
NUM_EPOCHS = 20
print("Starting training...")
for epoch in range(NUM_EPOCHS):
    model.train()
    TOTAL_LOSS = 0
    for batch in dataloader:
        corrupted_inputs, targets = batch
        corrupted_inputs = corrupted_inputs.to(device)
        targets = targets.to(device)

        # Forward pass
        outputs = model(corrupted_inputs)
        loss = criterion(outputs, targets)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        TOTAL_LOSS += loss.item() * corrupted_inputs.size(0)

    avg_loss = TOTAL_LOSS / len(dataset)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.6f}")

print("Training completed.")

# Save the reduced embeddings
model.eval()
with torch.no_grad():
    EMBEDDINGS_TENSOR = torch.FloatTensor(embeddings_loaded).to(device)
    reduced_embeddings = model.encoder(EMBEDDINGS_TENSOR).cpu().numpy()

np.save("data/reduced_embeddings.npy", reduced_embeddings)
print("Reduced embeddings saved to 'data/reduced_embeddings.npy'.")

torch.save(model.state_dict(), "data/denoising_autoencoder.pth")
print("Model saved to 'data/denoising_autoencoder.pth'.")
