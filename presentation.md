---
marp: true
theme: default
paginate: true
---

# **Lextral**
### Contract Clause Classification  
_Mistral Use-case Take Home_

---

## **1. Context & Problem**
- Contracts are long, complex, and time-consuming to review.
- Manual clause classification is error-prone and expensive.
- Automating this process can:
  - Reduce review time from minutes to seconds.
  - Improve consistency and compliance.
- **Dataset**: [LexGLUE / LEDGAR](https://huggingface.co/datasets/lex_glue)  
  - ~60,000 clauses  
  - 12,000 original categories → consolidated into 100 categories.

---

<!-- image: context.png -->
![bg right:40% 80%](images/context.png)

---

## **Target Users & Business Impact**
- **Target users:**
  - In-house counsels
  - Law firms
  - Compliance officers
- **Impact:**
  - Faster contract review
  - Reduced risk of missing critical clauses
  - Lower legal costs

---

<!-- image: impact.png -->
![bg right:40% 80%](images/impact.png)

---

## **2. Technical Overview**
Lextral has two main components:
1. **Experiments**  
   - Evaluate different classification strategies
   - Fine-tune models
2. **Backend**  
   - API (FastAPI) and web UI for end users  
   - Deployed on Kubernetes (Helm) with PostgreSQL + pgvector

---

<!-- image: architecture.png -->
![bg right:50% 80%](images/architecture.png)

---

## **3. Strategies Explored**
### **1️⃣ Zero-shot Prompt**
- Use a general-purpose LLM (*mistral-small*) with a simple prompt listing categories.
- Pros:
  - Quick to implement
  - No training needed
- Cons:
  - Limited domain adaptation
  - Moderate accuracy

---

### **2️⃣ Retrieval-Augmented Classification (RAG)**
- Embed all training clauses with **Mistral Embed**.
- Store embeddings in PostgreSQL + pgvector (HNSW index).
- At inference:
  - Embed the query clause
  - Retrieve top-*k* similar clauses and labels
  - Add them as few-shot examples in the prompt
- Pros:
  - Better accuracy than zero-shot
- Cons:
  - Higher latency due to retrieval

---

<!-- image: rag_diagram.png -->
![bg right:50% 80%](images/rag_diagram.png)

---

### **3️⃣ Fine-tuned Classifier**
- Fine-tune **Ministral 3B** with a classification head using LoRA.
- Training set: 60k clauses (consolidated categories).
- Pros:
  - Best accuracy
  - Low inference latency
- Cons:
  - Requires fine-tuning infrastructure

---

<!-- image: finetune.png -->
![bg right:50% 80%](images/finetune.png)

---

## **4. Evaluation Metrics**
- **Metrics used:**
  - Accuracy
  - Macro F1-score
  - Inference latency
  - Inference cost

---

## **Results Comparison**
| Strategy     | Accuracy | Macro F1 | Latency | Cost |
|--------------|----------|----------|---------|------|
| Zero-shot    | 60%      | 0.55     | 2.5 s   | $$   |
| RAG          | 72%      | 0.69     | 3.2 s   | $$$  |
| Fine-tune    | **82%**  | **0.79** | **0.9 s** | $    |

---

<!-- image: results_chart.png -->
![bg right:50% 80%](images/results_chart.png)

---

## **Analysis**
- **Fine-tuned model** outperforms all others in both accuracy and latency.
- **RAG** is a good compromise when fine-tuning is not possible.
- **Zero-shot** works for quick POCs but is limited for production.

---

## **5. Deployment**
- **Backend:** FastAPI
- **Database:** PostgreSQL + pgvector
- **Deployment:** Kubernetes + Helm
- **Access:**
  - Web UI: https://lextral.delavande.fr
  - API docs: https://lextral.delavande.fr/docs

---

<!-- image: deployment_diagram.png -->
![bg right:50% 80%](images/deployment_diagram.png)

---

## **6. Live Demo**
- **Scenario:**
  - Paste a clause in the UI
  - Model predicts its category instantly
- **API example:**
```bash
curl -X POST \
  https://lextral.delavande.fr/predict_finetuned \
  -H "Content-Type: application/json" \
  -d '{"texts": ["This Agreement may be amended..."]}'
````

---

## **7. Business Impact**

* **Legal teams**:

  * Faster reviews
  * Reduced errors
* **Organizations**:

  * Cost savings
  * Improved compliance
* **Scalability**:

  * Multi-language
  * Adaptable to other domains (insurance, banking, compliance)

---

<!-- image: business_impact.png -->

![bg right:40% 80%](images/business_impact.png)

---

## **8. Next Steps**

* Multi-language support
* Continual learning via user feedback in UI
* Clause extraction + classification
* Integration with contract management systems

---

<!-- image: roadmap.png -->

![bg right:40% 80%](images/roadmap.png)

---

# **Thank You**

📧 [julien@delavande.fr](mailto:julien@delavande.fr)
🔗 [https://lextral.delavande.fr](https://lextral.delavande.fr)
💻 [https://github.com/juliendelavande/lextral](https://github.com/juliendelavande/lextral)