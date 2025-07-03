
from sentence_transformers import SentenceTransformer, util
embedding_url = "https://api-inference.huggingface.co/pipeline/feature-extraction/sentence-transformers/all-MiniLM-L6-v2"
#embedding_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-MiniLM-L6-v2"
llm_url = "https://api-inference.huggingface.co/models/google/flan-t5-base"
#   llm_url = "https://api-inference.huggingface.co/models/mistralai/Mistral-7B-Instruct-v0.2"

#model_repo_id= "sentence-transformers/all-MiniLM-L6-v2"

model_repo_id= "google/flan-t5-base"

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

