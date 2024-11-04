import pickle

class JobEmbeddingGenerator:
    def __init__(self, client, batch_size=500):
        self.client = client
        self.batch_size = batch_size

    def generate_embeddings(self, df):
        embeddings_dict = {}

        # Process in batches to handle large datasets
        for i in range(0, len(df), self.batch_size):
            print(f"Processing batch {i // self.batch_size + 1}...")
            batch_df = df.iloc[i : i + self.batch_size]
            text_list = batch_df["combined_text"].tolist()
            ids = batch_df["job_id"].tolist()

            # Generate embeddings for the current batch
            embeddings = self.client.embeddings.create(
                input=text_list, model="text-embedding-3-large"
            ).data

            # Save embeddings by job id in a dictionary
            for job_id, response_embedding in zip(ids, embeddings):
                embeddings_dict[job_id] = response_embedding.embedding

        return embeddings_dict

    def save_embeddings(self, embeddings_dict, file_path="data/embeddings.pkl"):
        print(f"Size of embeddings_dict: {len(embeddings_dict)}")
        # Save the embeddings dictionary to a pickle file
        with open(file_path, "wb") as f:
            pickle.dump(embeddings_dict, f)
        print(f"Embeddings saved to {file_path}")
