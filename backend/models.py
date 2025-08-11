from typing import List, Optional
from pydantic import BaseModel

class ClauseRequest(BaseModel):
    texts: List[str]

class PredictionResponse(BaseModel):
    predicted_classes: List[str]

class RAGRequest(BaseModel):
    texts: List[str]
    top_k: int = 5
    min_sim: Optional[float] = None
    
class PredictionResponse(BaseModel):
    predicted_classes: List[str]
    prompts: Optional[List[str]] = None

