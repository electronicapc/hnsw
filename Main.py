from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
import hnswlib
from typing import List

app = FastAPI()


class VectorData(BaseModel):
    id: str
    vector: List[float]

class ANNRequest(BaseModel):
    dataset: List[VectorData]
    query: List[VectorData]
    k: int = 5

#Response
class Neighbor(BaseModel):
    id: str
    distance: float

class QueryResult(BaseModel):
    query_id: str
    neighbors: List[Neighbor]

class SearchResponse(BaseModel):
    message: str
    results: List[QueryResult]

@app.post("/search/", response_model=SearchResponse)
def search(request: ANNRequest):
    try:
        dataset_ids = [item.id for item in request.dataset]
        dataset_vectors = np.array([item.vector for item in request.dataset], dtype=np.float32)

        query_ids = [item.id for item in request.query]
        query_vectors = np.array([item.vector for item in request.query], dtype=np.float32)
        # Validate dimensions
        if dataset_vectors.shape[1] != query_vectors.shape[1]:
            raise HTTPException(
                status_code=400,
                detail="The dimensions of dataset vectors and query vectors do not match."
            )

        # # Initialize and build HNSW index
        dim = dataset_vectors.shape[1]
        index = hnswlib.Index(space='l2', dim=dim)
        index.init_index(max_elements=len(dataset_vectors), ef_construction=200, M=16)
        index.add_items(dataset_vectors, ids=np.arange(len(dataset_vectors)))
        index.set_ef(50)

        # Perform k-NN search
        labels, distances = index.knn_query(query_vectors, k=request.k)
        # Map results back to dataset IDs
        result = [
            {
                "query_id": query_ids[i],
                "neighbors": [
                    {"id": dataset_ids[label], "distance": distance}
                    for label, distance in zip(labels[i], distances[i])
                ],
            }
            for i in range(len(query_ids))
        ]
        #
        return {"message": "Search completed successfully.", "results": result}
    except Exception as e:
      raise HTTPException(status_code=500, detail=f"Internal error: {str(e)}")
