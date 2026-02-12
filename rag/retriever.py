import faiss


class FAISSRetriever:
    def __init__(self, embeddings):
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings)

    def search(self, query_embedding, top_k=3):
        _, indices = self.index.search(query_embedding, top_k)
        return indices[0]
