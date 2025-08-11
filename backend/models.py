from typing import List, Optional
from pydantic import BaseModel, Field

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

class RAGKNNRequest(BaseModel):
    texts: List[str]
    top_k: int = Field(5, ge=1, le=100)
    min_sim: Optional[float] = Field(None, description="Cosine similarity threshold in [0,1]")
    strategy: str = Field("weighted", pattern="^(weighted|softmax|majority)$")
    power: float = Field(2.0, ge=0.0, description="Used only for 'weighted' → sim**power")
    tau: float = Field(0.1, gt=0.0, description="Used only for 'softmax'")
    abstain_min_conf: float = Field(0.0, ge=0.0, le=1.0, description="If max prob < threshold → '__ABSTAIN__'")


