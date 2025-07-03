
from sentence_transformers import SentenceTransformer, util
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
#embedding_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
llm_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

