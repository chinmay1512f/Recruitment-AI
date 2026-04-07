from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

model = SentenceTransformer("all-MiniLM-L6-v2")

def match_resume(job_text, resume_text):
    if not job_text.strip() or not resume_text.strip():
        return 0.0

    embeddings = model.encode([job_text, resume_text])
    similarity = cosine_similarity(
        [embeddings[0]], [embeddings[1]]
    )[0][0]

    return round(similarity * 100, 2)
