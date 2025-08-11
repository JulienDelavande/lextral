import os
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from infer import predict_label, chat_complete
from rag.encoder_mistral import embed_texts
from rag.retriever_pgvector import search_similar
from models import ClauseRequest, PredictionResponse, RAGRequest, RAGKNNRequest
from mistral_utils import build_fewshot_prompt
from rag.utils import _score_neighbors, _normalize, _format_evidence
from typing import List

app = FastAPI(title="Clause Classifier API")

BASE_MODEL = os.getenv("MODEL_NAME", "mistral-small-latest")
FINE_TUNED_MODEL = os.getenv("FINE_TUNED_MODEL", "mistral-small-latest")

@app.get("/")
def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))

@app.post("/predict_base", response_model=PredictionResponse)
def predict_clause_base(request: ClauseRequest):
    try:
        prediction, messages = predict_label(
            texts=request.texts,
            model_name=BASE_MODEL,
            model_strategy="chatprompt"
        )
        print(f"Base model prediction: {prediction}")
        print(f"Messages: {messages}")
        prompts = [msg.content for msg in messages]
        return PredictionResponse(predicted_classes=prediction, prompts=prompts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_finetuned", response_model=PredictionResponse)
def predict_clause_finetuned(request: ClauseRequest):
    try:
        prediction, messages = predict_label(
            texts=request.texts,
            model_name=FINE_TUNED_MODEL,
            model_strategy="classifier"
        )
        print(f"Fine-tuned model prediction: {prediction}")
        print(f"Messages: {messages}")
        prompts = [msg.content for msg in messages]
        return PredictionResponse(predicted_classes=prediction, prompts=prompts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_rag", response_model=PredictionResponse)
def predict_clause_rag(req: RAGRequest):
    try:
        query_vecs = embed_texts(req.texts)
        print(f"query_vecs done")
        preds: list[str] = []
        prompts: list[str] = []
        for text, qv in zip(req.texts, query_vecs):
            neighbors = search_similar(qv, top_k=req.top_k, min_sim=req.min_sim)
            print(f"Found neighbors for text: {text}")
            messages = build_fewshot_prompt(text, neighbors)
            print(f"messages built : {messages}")
            pred = chat_complete(messages)
            print(f"pred done : {pred}")
            preds.append(pred)
            prompt = '\n'.join([msg.content for msg in messages])
            prompts.append(prompt)
        return PredictionResponse(predicted_classes=preds, prompts=prompts)
    except Exception as e:
        print(f"Error during RAG prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    
@app.post("/predict_rag_knn", response_model=PredictionResponse)
def predict_clause_rag_knn(req: RAGKNNRequest):
    try:
        if not req.texts:
            raise HTTPException(status_code=400, detail="texts is empty")

        query_vecs = embed_texts(req.texts)
        preds: List[str] = []
        evidences: List[str] = []
        neighbors_all = []

        for text, qv in zip(req.texts, query_vecs):
            neighbors = search_similar(
                qv,
                top_k=req.top_k,
                min_sim=req.min_sim,
                split="train"
            )
            if not neighbors:
                preds.append("__ABSTAIN__")
                evidences.append("No neighbors found.")
                continue

            raw_scores = _score_neighbors(
                neighbors,
                strategy=req.strategy,
                power=req.power,
                tau=req.tau
            )
            probs = _normalize(raw_scores)
            best_label, best_prob = max(probs.items(), key=lambda x: x[1]) if probs else ("__ABSTAIN__", 0.0)

            if best_prob < req.abstain_min_conf:
                best_label = "__ABSTAIN__"

            preds.append(best_label)
            evidences.append(_format_evidence(neighbors, probs))
            neighbors_all.append(neighbors)

        prompts = [f"Text: {text}\nEvidence:\n{evidence}\nNeighbors:\n{neighbors}" for text, evidence, neighbors in zip(req.texts, evidences, neighbors_all)]
        return PredictionResponse(predicted_classes=preds, prompts=prompts)

    except HTTPException:
        raise
    except Exception as e:
        print(f"Error during kNN RAG prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))