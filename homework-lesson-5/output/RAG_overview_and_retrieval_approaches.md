# Retrieval‑Augmented Generation (RAG)

## Overview
Retrieval‑augmented generation (RAG) couples a large language model (LLM) with an external **retrieval** component so that the model can look up and incorporate up‑to‑date information from a document store before generating its response [retrieval-augmented-generation.pdf, p. 1]. This architecture reduces hallucinations, lowers the need for costly model re‑training, and enables source attribution.

## Main Retrieval Approaches

### 1. Sparse (lexical) retrieval
* **Method** – Traditional inverted‑index search using term‑frequency statistics, most commonly **BM25**.
* **Characteristics** – Represents a document as a high‑dimensional one‑hot vector (dictionary length) that is mostly zeros. Matching is performed by dot‑product of query and document term vectors [retrieval-augmented-generation.pdf, p. 3‑4].
* **Pros/Cons** – Fast and well‑understood, but limited to exact word overlap and cannot capture semantic similarity.

### 2. Dense (neural) retrieval
* **Method** – Encode queries and passages into low‑dimensional dense embeddings (e.g., using BERT‑based encoders such as DPR, ANCE) and retrieve by nearest‑neighbor search in a vector database.
* **Characteristics** – Dense vectors capture meaning rather than exact word identity, allowing semantic matches even when terminology differs [retrieval-augmented-generation.pdf, p. 2‑3].
* **Pros/Cons** – Improves recall for paraphrased queries, but requires an embedding model and is computationally heavier than lexical search.

### 3. Hybrid retrieval
* **Method** – Combine sparse and dense representations—e.g., **hybrid vectors** that concatenate a sparse one‑hot term vector with a dense semantic vector, or fuse separate BM25 and dense scores at retrieval time.
* **Characteristics** – Leverages the efficiency of sparse dot‑product while retaining the semantic coverage of dense embeddings [retrieval-augmented-generation.pdf, p. 3].
* **Pros/Cons** – Often yields higher overall relevance than either method alone, at the cost of more complex indexing and scoring pipelines.

### 4. Reranking (optional refinement)
* **Method** – After an initial top‑k retrieval (using any of the above), a cross‑encoder or other learned model re‑scores the candidates to promote the most relevant documents.
* **Characteristics** – Reranking improves precision by jointly encoding query‑document pairs, but is applied only to a small set of candidates to keep latency reasonable [retrieval-augmented-generation.pdf, p. 3‑4].

## Sources
- **retrieval-augmented-generation.pdf** – Primary overview of RAG, definitions, and discussion of sparse vs. dense vs. hybrid vectors (pages 1‑4). 
- **retrieval-augmented-generation.pdf** – Mention of hybrid vector approaches that combine dense and sparse representations (page 3).
