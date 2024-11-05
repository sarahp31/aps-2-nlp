import time
import logging
import numpy as np
import torch
import torch.nn as nn
from openai import OpenAI
from app.sqlite3db import SQLPostingDatabase

logger = logging.getLogger(__name__)

client = OpenAI()

# Initialize the database driver
driver = SQLPostingDatabase()
# Load the refined embeddings
reduced_embeddings = np.load("data/reduced_embeddings.npy")


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


# Function to generate embedding for the query
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
    logger.info("Generating embedding for the query...")
    query_embedding = generate_query_embedding(query)

    # Convert to torch tensor
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    query_embedding_tensor = torch.FloatTensor(query_embedding).to(device)

    # Load the trained autoencoder model
    embedding_size = reduced_embeddings.shape[1]

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
    model = DenoisingAutoencoder(input_size=3072).to(device)
    model.load_state_dict(
        torch.load("data/denoising_autoencoder.pth", map_location=device)
    )
    model.eval()
    logger.info("Model loaded successfully.")

    # Encode the query embedding using the encoder
    with torch.no_grad():
        query_embedding_tensor = query_embedding_tensor.unsqueeze(
            0
        )  # Add batch dimension
        query_encoded = (
            model.encoder(query_embedding_tensor).cpu().numpy()
        )  # Shape: (1, 512)

    # Compute cosine similarities
    logger.info("Computing cosine similarities...")

    # Normalize embeddings
    def normalize_embeddings(embeddings):
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        return embeddings / norms

    query_encoded_norm = query_encoded / np.linalg.norm(query_encoded)
    job_embeddings_norm = normalize_embeddings(reduced_embeddings)

    # Compute cosine similarity
    similarities = np.dot(
        job_embeddings_norm, query_encoded_norm.T
    ).squeeze()  # Shape: (num_jobs,)

    # Filter and get top N results
    indices = np.where(similarities >= threshold)[0]
    if len(indices) == 0:
        return []

    # Get top N indices based on similarity scores
    top_indices = similarities.argsort()[::-1]  # Descending order
    top_indices = top_indices[:top_n]

    # Prepare the results
    results = []
    for idx in top_indices:
        job = {
            "job_id": data["job_id"][idx],
            "job_title": data["job_title"][idx],
            "job_description": data["job_description"][idx],
            "similarity": float(similarities[idx]),
        }
        results.append(job)

    end_time = time.time()
    logger.info(f"Query processed in {end_time - start_time:.2f} seconds.")

    return {
        "time_taken": end_time - start_time,
        "message": "OK",
        "results": results,
    }
