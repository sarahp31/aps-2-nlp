import time
import logging
import functools
import numpy as np
import faiss
import torch
import torch.nn as nn
from openai import OpenAI
from app.sqlite3db import SQLPostingDatabase

faiss.omp_set_num_threads(1)

logger = logging.getLogger(__name__)

client = OpenAI()

# Initialize the database driver
driver = SQLPostingDatabase()

# Load the refined embeddings
reduced_embeddings = np.load("data/reduced_embeddings.npy").astype("float32")


# Load job data (titles and descriptions)
def get_jobs_data():
    jobs = driver.get_all()
    payload = {"job_id": [], "job_title": [], "job_description": []}
    for job in jobs:
        payload["job_id"].append(job["id"])
        payload["job_title"].append(job["title"])
        payload["job_description"].append(job["description"])
    return payload


data = get_jobs_data()


# Define the DenoisingAutoencoder class
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


# Initialize the model and load weights
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DenoisingAutoencoder(input_size=3072).to(device)
model.load_state_dict(torch.load("data/denoising_autoencoder.pth", map_location=device))
model.eval()
logger.info("Model loaded successfully.")


# Precompute normalized job embeddings
def normalize_embeddings(embeddings):
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / norms


job_embeddings_norm = normalize_embeddings(reduced_embeddings)

# Build FAISS index for similarity search
d = reduced_embeddings.shape[1]
index = faiss.IndexFlatIP(d)
index.add(job_embeddings_norm)


@functools.lru_cache(maxsize=128)
def generate_query_embedding(query_text):
    response = client.embeddings.create(
        input=[query_text], model="text-embedding-3-large"
    )
    query_embedding = response.data[0].embedding
    query_embedding = np.array(query_embedding)
    return query_embedding


# Function to process the query and return ranked jobs
def get_query(query: str, threshold=0.33, top_n=10):
    start_time = time.time()
    # Generate embedding for the query
    query_embedding = generate_query_embedding(query)

    # Convert to torch tensor
    query_embedding_tensor = torch.FloatTensor(query_embedding).to(device)

    # Encode the query embedding using the encoder
    with torch.no_grad():
        query_embedding_tensor = query_embedding_tensor.unsqueeze(
            0
        )  # Add batch dimension
        query_encoded = (
            model.encoder(query_embedding_tensor).cpu().numpy()
        )  # Shape: (1, 512)

    # Normalize the query encoded vector
    query_encoded_norm = query_encoded / np.linalg.norm(query_encoded)
    query_encoded_norm = query_encoded_norm.astype("float32")

    # Compute cosine similarities using FAISS index
    k = min(top_n * 5, job_embeddings_norm.shape[0])  # Retrieve more than needed
    D, I = index.search(query_encoded_norm, k)

    # D is similarities, I is indices
    similarities = D.flatten()
    indices = I.flatten()

    # Filter out similarities below threshold
    valid = similarities >= threshold
    similarities = similarities[valid]
    indices = indices[valid]

    if len(indices) == 0:
        return []

    # Get top N results
    sorted_indices = np.argsort(-similarities)  # Sort in descending order
    top_indices = indices[sorted_indices][:top_n]
    top_similarities = similarities[sorted_indices][:top_n]

    # Prepare the results
    results = []
    for idx, sim in zip(top_indices, top_similarities):
        job = {
            "job_id": data["job_id"][idx],
            "job_title": data["job_title"][idx],
            "job_description": data["job_description"][idx],
            "similarity": float(sim),
        }
        results.append(job)

    end_time = time.time()
    logger.info(f"Query processed in {end_time - start_time:.2f} seconds.")

    return {
        "time_taken": end_time - start_time,
        "message": "OK",
        "results": results,
    }
