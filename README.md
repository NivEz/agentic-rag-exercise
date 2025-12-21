# Agentic RAG System

A Retrieval-Augmented Generation (RAG) system that uses hierarchical chunking, summarization, and intelligent routing to answer questions from insurance claim documents.

## Overview

This system processes PDF documents through multiple pipelines, stores them in ChromaDB, and uses LangChain agents with intelligent routing to retrieve and answer questions. The system is evaluated using RAGAS metrics.

## Architecture

The system consists of four main components that work together:

`PDF Document => Data Ingestion => ChromaDB Storage => Retrieval => Agents => Evaluation`

## Components

### 1. Data Ingestion

The ingestion process uses **LlamaIndex** to process PDFs and stores them in **ChromaDB** using two pipelines:

#### Hierarchical Chunks Pipeline

-   Processes documents using HierarchicalNodeParser to create chunks at multiple granularity levels (large, medium, small)
-   Stores hierarchical chunks in the chunks collection in ChromaDB
-   Only leaf nodes are stored in the vector store; all nodes (including parent nodes) are stored in the docstore for auto-merging retrieval

#### Summaries Pipeline

-   Generates summaries using MapReduce approach with SentenceSplitter
-   Creates two types of summaries:
    -   **Chunk-level summaries**: Summaries for individual document sections
    -   **Document-level summaries**: High-level overview summaries for entire documents
-   Stores summaries in the summaries collection in ChromaDB

Both pipelines run during ingestion, creating two separate vector store collections, optimized for different query types.

### 2. Retrieval

The system uses two retrieval strategies:

#### Summary Index Retrieval

-   Uses SummariesRetriever to search the summaries collection
-   Optimized for high-level, overview questions
-   Returns document-level or chunk-level summaries based on query specificity
-   Optimized for main key events, timeline and general information

#### Chunks Index Retrieval (Auto-Merging)

-   Uses AutoMergingRetriever to search the chunks collection
-   Retrieves leaf nodes and automatically merges them back into parent nodes when multiple siblings are retrieved
-   Optimized for specific, detailed questions requiring exact information extraction
-   Leverages the hierarchical structure stored in the docstore

### 3. Agents

Built with **LangChain**, the system uses three types of agents:

#### Query Router Agent

-   Analyzes user queries and determines the appropriate retrieval strategy
-   Routes queries to either chunks (auto-merging) or summaries retrieval
-   Uses LLM-based decision making to select the best strategy

#### Needle Agent

-   Handles specific, detailed queries
-   Uses AutoMergingRetriever to find precise information
-   Best for questions requiring exact facts, numbers, or specific details
-   If did not retrieve enough context, the agent will reconstruct the query

#### Summary Agent

-   Handles broad, overview queries
-   Uses SummariesRetriever to retrieve high-level information
-   Has two tools:
    -   retrieve_context: For document-level summaries (very broad questions)
    -   retrieve_detailed_context: For chunk-level summaries (specific but high-level questions)
-   If did not retrieve enough context, the agent will reconstruct the query

All agents use LangChain's agent framework with tool calling to retrieve context and generate answers.

### 4. Evaluation

The system uses **RAGAS** (Retrieval-Augmented Generation Assessment) to evaluate performance:

-   **Metrics**: answer correctness, faithfulness, context precision
-   **Dataset**: Pre-defined questions with ground truth answers
-   **Process**:
    1. Query Router Agent processes each question
    2. Retrieves contexts and generates answers
    3. RAGAS evaluates answers against ground truth
    4. Results saved to CSV files with timestamps

## Setup

### 1. Create Virtual Environment

```bash
python -m venv venv
```

### 2. Activate Virtual Environment

**Windows (PowerShell):**

```powershell
.\venv\Scripts\Activate.ps1
```

**Windows (Command Prompt):**

```cmd
venv\Scripts\activate.bat
```

**Linux/Mac:**

```bash
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Environment Variables

Create a `.env` file in the project root with your OpenAI API key:

```
OPENAI_API_KEY=your_api_key_here
```

## Usage

### Data Ingestion

`bash
python src/data_ingestion/pipeline.py --pdf "path/to/file.pdf"
`

### Query Processing

`python
from src.agents.query_router import QueryRouterAgent

router = QueryRouterAgent()
response = router.answer_with_contexts("What is the claim number?")
`

### Evaluation

`ash
python evaluations/main.py
`

## Configuration

All settings are configured in config/config.yaml, including:

-   LLM settings (model, temperature)
-   Chunking parameters (sizes, overlap)
-   Vector store settings (collections, embeddings)
-   Summarization settings

## Dependencies

-   **LlamaIndex**: Document processing and hierarchical chunking
-   **ChromaDB**: Vector storage
-   **LangChain**: Agent framework
-   **RAGAS**: Evaluation metrics
-   **OpenAI**: LLM and embeddings
