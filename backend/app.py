from fastapi import FastAPI
from backend.retrieval.vector_store import load_documents, create_vector_store, search

app = FastAPI()

docs = load_documents("data/course_materials")
vector_store = create_vector_store(docs)

@app.get("/query")
def query_course(q: str):
    results = search(q, vector_store)
    return {"results": results}
