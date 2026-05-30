# RAG-Based Conversational Platform for Domain-Specific Knowledge Retrieval

A Retrieval-Augmented Generation (RAG) platform designed for enterprise document intelligence and domain-specific knowledge retrieval.

The system enables semantic search across large document collections and generates grounded responses using multiple Large Language Models (LLMs). In addition to the baseline RAG workflow, the platform includes conversational memory for handling follow-up questions and supports benchmarking of multiple LLM APIs under different retrieval configurations.

---

## Features

- Semantic document retrieval using SentenceTransformers and FAISS
- Multi-LLM response generation
- Source-aware answer generation
- Conversational memory for follow-up questions
- Query rewriting for context-aware retrieval
- Interactive Gradio user interface
- Quantitative and qualitative benchmarking framework

---

## System Architecture

<p align="center">
  <img src="https://github.com/user-attachments/assets/d5b59b1d-1ac9-4073-84f0-9d21caaf1316" width="450">
</p>

<p align="center">
<i>Baseline RAG Architecture</i>
</p>

### Workflow

1. Documents are processed and converted into embeddings.
2. Embeddings are stored in a FAISS vector database.
3. User queries are embedded into the same vector space.
4. Top-k relevant document chunks are retrieved.
5. Retrieved chunks are combined into a context.
6. The context is passed to an LLM for answer generation.
7. Generated answers are returned together with source references.

---

## Models Evaluated

The platform was benchmarked using multiple LLM APIs:

| Model |
|---------|
| Gemma |
| Pixtral |
| Qwen |
| InternVL |

---

## Retrieval Configurations

Multiple retrieval and generation configurations were evaluated.

| Configuration | Top-k | Temperature | Max Tokens |
|--------------|-------|------------|------------|
| C1 | 7 | 0.2 | 768 |
| C2 | 10 | 0.3 | 1024 |
| C3 | 6 | 0.1 | 512 |
| C4 | 10 | 0.4 | 2048 |

---

## Evaluation Metrics

Responses were evaluated using:

### Semantic Similarity
Measures similarity in meaning between generated responses and reference responses.

### BLEU
Measures overlap of words and phrases between responses.

### ROUGE-L
Measures overlap of important content and sequence structure.

### Manual Evaluation
Responses were manually reviewed for:

- Technical depth
- Semantic alignment
- Response structure
- Consistency
- Grounding and caution

---

## Benchmarking Highlights

Key observations from the benchmarking study:

- Qwen achieved the strongest semantic alignment overall.
- Gemma consistently produced detailed technical explanations.
- Larger retrieval depth generally improved response quality.
- Retrieval configuration C4 produced the strongest overall semantic similarity.
- Multiple models demonstrated competitive performance across different question categories.

---

## Conversational RAG Extension

The platform was extended with conversational memory to support multi-turn interactions.

### Added Components

- Conversation history management
- Follow-up question handling
- Query rewriting
- Context-aware retrieval

Example:

**User Question**

```
What is Grounded DINO?
```

**Follow-up Question**

```
What are its limitations?
```

**Rewritten Retrieval Query**

```
What are the limitations of Grounded DINO?
```

This allows the retriever to access relevant information even when follow-up questions contain ambiguous references such as "it", "they", or "those methods".

---

## Technology Stack

### Retrieval

- FAISS
- SentenceTransformers
- Vector Similarity Search

### Backend

- Python
- Pandas
- NumPy

### Interface

- Gradio

### LLM Integration

- Enterprise LLM APIs
- Prompt Engineering
- Context-Aware Retrieval

---

## Repository Structure

```text
.
├── app_gradio.py
├── metrics.py
├── embeddings/
├── faiss_index/
├── data/
└── README.md
```

---

## Future Improvements

- Hybrid vector and metadata-based retrieval
- Semantic and overlap-aware chunking strategies
- Rich metadata generation for retrieved chunks
- Advanced agentic retrieval workflows
- Multi-agent reasoning systems
- Automated retrieval validation

---

## Disclaimer

This repository demonstrates a document intelligence and conversational RAG workflow for research and engineering purposes. No proprietary datasets, internal document collections, or confidential enterprise information are included.
