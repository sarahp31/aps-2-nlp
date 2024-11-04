import pickle
from sklearn.metrics.pairwise import cosine_similarity

def get_query(query: str, threshold=0.33, top_n=10):
    with open('data/tfidf_model.pkl', 'rb') as f:
        vectorizer, tfidf_matrix, df = pickle.load(f)
    
    query_tfidf = vectorizer.transform([query])
    
    cosine_similarities = cosine_similarity(query_tfidf, tfidf_matrix).flatten()
    
    result = []
    for i, similarity in enumerate(cosine_similarities):
        if similarity >= threshold:
            job_data = {
                'title': df.iloc[i]['job_title'],
                'content': df.iloc[i]['job_description'],
                'relevance': round(similarity, 3)
            }
            result.append(job_data)
    
    result = sorted(result, key=lambda x: x['relevance'], reverse=True)
    
    if top_n:
        result = result[:top_n]

    return {
        'results': result,
        'message': 'OK',
    }