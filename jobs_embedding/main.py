import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from app.services import get_jobs_data
from app.JobEmbeddingGenerator import JobEmbeddingGenerator

# Load environment variables
load_dotenv()

# Initialize OpenAI client
client = OpenAI()

# Fetch job data
data = get_jobs_data()

# Convert to DataFrame
df = pd.DataFrame(data)
df["combined_text"] = (
    "Title: " + df["job_title"] + "\nDescription: " + df["job_description"]
)

print(f"Dataframe shape: {df.shape}")

# Usage example
generator = JobEmbeddingGenerator(client)
embeddings_dict = generator.generate_embeddings(df)
generator.save_embeddings(embeddings_dict)
