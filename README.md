# Insurance Claim Timeline Retrieval System

## Project Overview

This project implements a multi-agent GenAI system for managing and querying insurance claim timelines. The system uses hierarchical data indexing, multiple retrieval strategies, and agent-based orchestration to handle both high-level summary queries and precise factual lookups.

## Table of Contents

1. [Architecture Overview](#architecture-overview)
2. [Design Choices & Rationale](#design-choices--rationale)
3. [Tool Selection](#tool-selection)
4. [Data Structure & Indexing](#data-structure--indexing)
5. [Agent Design](#agent-design)
6. [Implementation Strategy](#implementation-strategy)
7. [Chunking Strategy](#chunking-strategy)
8. [Recall Improvement](#recall-improvement)

---

## Architecture Overview

The system follows a two-pipeline architecture:

### 1. RAG Pipeline (Data Ingestion & Indexing)

-   **File Import**: Accepts insurance claim documents (PDF format)
-   **Structure Identification**: Uses LLM to identify document structure (sections, subsections)
-   **Hierarchical Chunking**: Creates three levels of chunk granularity (small/medium/large)
-   **Summary Generation**: Builds a Summary Index using MapReduce summarization
-   **Vectorization**: Stores embeddings in vector database

### 2. Agentic Pipeline (Query Processing)

-   **User Input**: Receives natural language queries
-   **Router Agent**: Analyzes query intent and routes to appropriate agent
-   **Specialist Agents**:
    -   Summarization Expert Agent (uses Summary Index)
    -   Needle-in-a-Haystack Agent (uses Hierarchical Index)
-   **Retrieval & Response**: Agents retrieve relevant context and generate answers

---

## Design Choices & Rationale

### Why Hierarchical Structure?

**Claim → Document → Section → Chunk** hierarchy enables:

1. **Selective Retrieval**: Agents can navigate to specific document sections without scanning entire claims
2. **Metadata Filtering**: Filter by claim ID, document type, section name, etc.
3. **Context Preservation**: Maintains relationships between chunks, allowing reconstruction of broader context
4. **Query-Type Alignment**: Different query types match different granularity levels
    - Timeline questions → Document/Section level
    - Specific facts → Chunk level
    - Overview questions → Summary level

### Why Multiple Chunk Sizes?

The **Auto-Merging Retriever** pattern requires multiple granularities:

-   **Small Chunks (256 tokens)**:
    -   High precision for exact fact retrieval
    -   Minimal noise in search results
    -   Best for "needle-in-haystack" queries
-   **Medium Chunks (512 tokens)**:
    -   Balanced precision and context
    -   Handles queries requiring moderate context
    -   Good default for most factual queries
-   **Large Chunks (1024 tokens)**:
    -   Maximum context for complex reasoning
    -   Preserves narrative flow
    -   Used when small chunks lack sufficient context

**Auto-Merging Logic**: Start with small chunks → if confidence is low or context insufficient → merge upward to medium → then large.

### Why Summary Index?

The Summary Index serves multiple purposes:

1. **Fast High-Level Queries**: Timeline and overview questions don't require deep document traversal
2. **Efficient Storage**: Summaries are compact compared to full document retrieval
3. **MapReduce Strategy**:
    - **Map Phase**: Summarize each chunk independently
    - **Reduce Phase**: Combine chunk summaries into section summaries, then document summaries
    - Enables parallel processing and maintains hierarchy

---

## Tool Selection

### Framework: **Hybrid Approach Recommended**

#### Option 1: **LlamaIndex for Indexing + LangChain for Agents** (Recommended)

**Rationale for Hybrid Approach:**

**LlamaIndex for Data Indexing:**

-   **Built-in Hierarchical Support**: Native support for hierarchical indexing (`HierarchicalNodeParser`, `ParentChildRetriever`)
-   **Summary Index**: Includes `SummaryIndex` class with MapReduce summarization out-of-the-box
-   **Auto-Merging Retriever**: Implements `AutoMergingRetriever` pattern natively
-   **Metadata Management**: Strong metadata filtering and node relationship tracking
-   **Simpler Indexing API**: Less boilerplate for building indexes

**LangChain for Agent Orchestration:**

-   **Mature Agent Framework**: More sophisticated agent patterns (ReAct, Plan-and-Execute, AgentExecutor)
-   **Better Tool Integration**: Excellent support for custom tools and MCP (Model Context Protocol) integration
-   **Flexible Routing**: More control over agent routing logic and decision-making
-   **Multi-Agent Orchestration**: Better support for complex multi-agent workflows
-   **Function Calling**: Superior structured outputs and tool use patterns
-   **Agent Memory**: Built-in conversation memory and state management
-   **Error Handling**: More robust error handling and retry mechanisms

**Why This Hybrid Works:**

-   Use LlamaIndex's strengths for building and managing indexes
-   Use LangChain's strengths for agent logic, routing, and tool integration
-   Best of both worlds: simple indexing + powerful agent framework

#### Option 2: **Pure LangChain** (Alternative)

**Advantages:**

-   **Unified Framework**: Single framework for entire pipeline
-   **More Control**: Full control over indexing and retrieval logic
-   **Better MCP Support**: Native MCP tool integration
-   **Customizable**: Easier to customize retrieval strategies

**Disadvantages:**

-   **More Boilerplate**: Need to implement hierarchical chunking manually
-   **No Built-in Summary Index**: Must build MapReduce summarization from scratch
-   **More Complex**: Requires more code for indexing pipeline

#### Option 3: **Pure LlamaIndex** (Simpler but Less Flexible)

**Advantages:**

-   **Simpler**: Less code overall, built-in patterns
-   **Faster Development**: Quick to get started
-   **Integrated**: Everything in one framework

**Disadvantages:**

-   **Less Flexible**: Agent framework less mature than LangChain
-   **Limited MCP Support**: Weaker MCP tool integration
-   **Less Control**: Less control over agent behavior and routing

### Recommendation: **Hybrid (LlamaIndex + LangChain)**

For this assignment, we recommend:

-   **LlamaIndex**: Use for building hierarchical indexes, Summary Index, and Auto-Merging Retriever
-   **LangChain**: Use for implementing agents (Router, Summarization Expert, Needle-in-Haystack) with MCP tool integration

This approach leverages each framework's strengths while meeting all assignment requirements.

### Vector Database: **ChromaDB** (Recommended)

**Rationale:**

-   **Simplicity**: Pure Python, no external dependencies (unlike Pinecone, Weaviate)
-   **Local-First**: Runs entirely locally, perfect for development and testing
-   **Metadata Filtering**: Strong support for filtering by metadata (claim_id, document_type, section, chunk_size)
-   **Lightweight**: Minimal setup, no Docker required
-   **Persistent Storage**: Can persist to disk for production use
-   **LlamaIndex Integration**: Native `ChromaVectorStore` integration

**Alternatives Considered:**

-   **FAISS**: Fast but in-memory only, no metadata filtering
-   **Qdrant**: More complex setup, requires Docker
-   **Pinecone**: Cloud-based, requires API keys, overkill for this exercise

### LLM Provider: **OpenAI** (via API)

**Rationale:**

-   **Function Calling**: Excellent support for structured outputs (model prompts as functions)
-   **Consistency**: Reliable for agent routing decisions
-   **Cost-Effective**: GPT-3.5-turbo sufficient for routing, GPT-4 for complex summarization

---

## Data Structure & Indexing

### Hierarchical Data Model

```
Claim (Top Level)
├── Metadata: claim_id, date, status
│
├── Document 1
│   ├── Metadata: doc_type, date, page_range
│   │
│   ├── Section 1: "Initial Report"
│   │   ├── Summary (Summary Index entry)
│   │   ├── Large Chunk 1 (1024 tokens)
│   │   │   ├── Medium Chunk 1.1 (512 tokens)
│   │   │   │   ├── Small Chunk 1.1.1 (256 tokens)
│   │   │   │   └── Small Chunk 1.1.2 (256 tokens)
│   │   │   └── Medium Chunk 1.2 (512 tokens)
│   │   └── Large Chunk 2 (1024 tokens)
│   │
│   └── Section 2: "Medical Records"
│       └── [Similar structure]
│
└── Document 2
    └── [Similar structure]
```

### Index Types

#### 1. Summary Index

-   **Structure**: Flat index of summaries
-   **Content**: Document summaries, section summaries, timeline overviews
-   **Metadata**: `claim_id`, `document_id`, `section_id`, `summary_type`, `timestamp`
-   **Use Case**: High-level queries, timeline reconstruction

#### 2. Hierarchical Chunk Index

-   **Structure**: Multi-level index with parent-child relationships
-   **Content**: Small, medium, and large chunks with relationships
-   **Metadata**: `claim_id`, `document_id`, `section_id`, `chunk_size`, `parent_id`, `child_ids`, `chunk_index`
-   **Use Case**: Precise factual queries, deep document search

### Vector Store Schema

**Summaries Table:**

```python
{
    "id": "summary_<claim_id>_<doc_id>_<section_id>",
    "embedding": [vector],
    "text": "Summary text...",
    "metadata": {
        "claim_id": str,
        "document_id": str,
        "section_id": str,
        "summary_type": "document" | "section" | "timeline",
        "timestamp": datetime,
        "source_chunks": [chunk_ids]
    }
}
```

**Chunks Table:**

```python
{
    "id": "chunk_<claim_id>_<doc_id>_<section_id>_<size>_<index>",
    "embedding": [vector],
    "text": "Chunk text...",
    "metadata": {
        "claim_id": str,
        "document_id": str,
        "section_id": str,
        "chunk_size": "small" | "medium" | "large",
        "parent_id": str | None,
        "child_ids": [str],
        "chunk_index": int,
        "start_char": int,
        "end_char": int
    }
}
```

---

## Agent Design

### 1. Manager (Router) Agent

**Responsibilities:**

-   Analyze user query intent
-   Determine query type (summary vs. factual)
-   Select appropriate agent and index
-   Route query to specialist agent

**Implementation (LangChain):**

-   Uses LangChain `AgentExecutor` with `create_openai_tools_agent`
-   Custom tool: `classify_query` using OpenAI function calling
-   Query classification categories:
    -   `TIMELINE`: "What happened on date X?", "Show me the timeline"
    -   `SUMMARY`: "Give me an overview", "What's the status?"
    -   `FACTUAL`: "What was the claim amount?", "Who was the adjuster?"
    -   `NEEDLE_IN_HAYSTACK`: "Find specific detail X in document Y"

**Routing Logic:**

```python
# LangChain agent with routing tools
tools = [
    classify_query_tool,
    route_to_summarization_agent_tool,
    route_to_needle_haystack_agent_tool
]

agent = create_openai_tools_agent(llm, tools, prompt)
executor = AgentExecutor(agent=agent, tools=tools, verbose=True)
```

**Tools:**

-   `classify_query`: LLM function calling tool for query classification
-   `route_to_summarization_agent`: Routes to Summarization Expert Agent
-   `route_to_needle_haystack_agent`: Routes to Needle-in-Haystack Agent

### 2. Summarization Expert Agent

**Responsibilities:**

-   Handle high-level and timeline queries
-   Query Summary Index efficiently
-   Synthesize information from multiple summaries
-   Generate timeline reconstructions

**Implementation (LangChain):**

-   Uses LangChain `AgentExecutor` with retrieval tools
-   Custom tool: `retrieve_from_summary_index` (wraps LlamaIndex Summary Index)
-   Uses LangChain's `RunnableLambda` for query optimization
-   Can aggregate multiple summaries using LangChain chains

**Query Construction:**

-   Tool: `extract_entities` - extracts dates, claim IDs, document types
-   Tool: `filter_by_metadata` - filters summaries by metadata
-   Tool: `retrieve_summaries` - retrieves top-K summaries from Summary Index
-   Tool: `aggregate_summaries` - combines multiple summaries

**Tools:**

-   `retrieve_from_summary_index`: Wraps LlamaIndex Summary Index retriever
-   `extract_entities`: Extracts key entities from query
-   `filter_by_metadata`: Filters summaries by metadata (claim_id, document_id, etc.)
-   `aggregate_summaries`: Combines multiple summaries into coherent response

### 3. Needle-in-a-Haystack Agent

**Responsibilities:**

-   Handle precise factual queries
-   Search deep within hierarchical index
-   Use Auto-Merging Retriever pattern
-   Navigate parent-child relationships

**Implementation (LangChain):**

-   Uses LangChain `AgentExecutor` with hierarchical retrieval tools
-   Custom tool: `retrieve_hierarchical_chunks` (wraps LlamaIndex Auto-Merging Retriever)
-   Uses LangChain's tool calling for adaptive retrieval strategy
-   Can traverse hierarchy using parent-child navigation tools

**Query Construction:**

-   Tool: `extract_search_entities` - extracts specific entities and facts
-   Tool: `filter_by_metadata` - filters by document_id, section_id when available
-   Tool: `retrieve_small_chunks` - retrieves initial small chunks
-   Tool: `merge_to_medium` - merges to parent medium chunks if needed
-   Tool: `merge_to_large` - merges to parent large chunks if needed

**Auto-Merging Logic (Implemented as LangChain Tools):**

1. Tool: `retrieve_small_chunks` - Retrieve top-K small chunks
2. Tool: `evaluate_context_sufficiency` - Check if context is sufficient
3. If confidence score < threshold OR context insufficient:
    - Tool: `merge_to_medium` - Merge to parent medium chunks
4. If still insufficient:
    - Tool: `merge_to_large` - Merge to parent large chunks
5. Return merged context

**Tools:**

-   `retrieve_hierarchical_chunks`: Wraps LlamaIndex Hierarchical Index retriever
-   `retrieve_small_chunks`: Retrieves small chunks (256 tokens)
-   `merge_to_medium`: Merges to medium chunks (512 tokens)
-   `merge_to_large`: Merges to large chunks (1024 tokens)
-   `navigate_hierarchy`: Navigates parent-child relationships
-   `filter_by_metadata`: Filters by document_id, section_id, claim_id
-   `evaluate_context_sufficiency`: Evaluates if retrieved context is sufficient

---

## Implementation Strategy

### Phase 1: Data Ingestion Pipeline

1. **PDF Processing**

    - Use `PyPDF2` or `pdfplumber` for text extraction
    - Preserve page numbers and basic structure

2. **Structure Identification**

    - Use LLM (GPT-4) to identify document structure
    - Prompt: "Identify sections, subsections, and their boundaries"
    - Output: Structured JSON with section names and boundaries

3. **Hierarchical Chunking**

    - Use LlamaIndex `HierarchicalNodeParser`
    - Configure chunk sizes: 256, 512, 1024 tokens
    - Set overlap: 50 tokens between chunks
    - Maintain parent-child relationships

4. **Summary Generation (MapReduce)**

    - **Map Phase**: Summarize each chunk independently
    - **Reduce Phase**:
        - Combine chunk summaries → section summaries
        - Combine section summaries → document summaries
    - Use LlamaIndex `SummaryIndex` with MapReduce mode

5. **Vectorization & Storage**
    - Generate embeddings using OpenAI `text-embedding-3-small`
    - Store in ChromaDB with metadata
    - Maintain separate collections: `summaries` and `chunks`

### Phase 2: Agent Implementation (Using LangChain)

1. **Router Agent**

    - Use LangChain `AgentExecutor` with `create_openai_tools_agent`
    - Implement query classifier using OpenAI function calling
    - Create custom tool: `classify_query` that returns query type
    - Use LangChain's routing logic to delegate to specialist agents
    - Handle edge cases (ambiguous queries) with fallback logic

2. **Summarization Expert Agent**

    - Use LangChain `AgentExecutor` with retrieval tools
    - Create custom tool: `retrieve_from_summary_index` (wraps LlamaIndex Summary Index)
    - Add query optimization tool: `extract_entities` for metadata filtering
    - Implement summary aggregation using LangChain's `RunnableLambda`
    - Use LangChain's memory for conversation context

3. **Needle-in-a-Haystack Agent**
    - Use LangChain `AgentExecutor` with hierarchical retrieval tools
    - Create custom tool: `retrieve_hierarchical_chunks` (wraps LlamaIndex Auto-Merging Retriever)
    - Add tool: `navigate_hierarchy` for parent-child traversal
    - Implement confidence scoring tool for merge decisions
    - Use LangChain's tool calling for adaptive retrieval strategy

### Phase 3: Integration & Testing

1. **End-to-End Flow**

    - Connect all components
    - Test with sample queries
    - Validate routing decisions

2. **Evaluation**
    - Implement LLM-as-a-Judge evaluation
    - Create test query set
    - Measure accuracy, recall, precision

---

## Chunking Strategy

### Chunk Sizes

-   **Small: 256 tokens**

    -   ~200 words
    -   Best for: Exact fact retrieval, specific details
    -   Overlap: 50 tokens (20% overlap)

-   **Medium: 512 tokens**

    -   ~400 words
    -   Best for: Balanced queries, moderate context
    -   Overlap: 50 tokens (10% overlap)

-   **Large: 1024 tokens**
    -   ~800 words
    -   Best for: Complex reasoning, narrative context
    -   Overlap: 100 tokens (10% overlap)

### Overlap Strategy

**Rationale:**

-   **50-token overlap** prevents information loss at chunk boundaries
-   Ensures context continuity when merging chunks
-   Prevents splitting of related concepts across chunks
-   Improves recall for queries spanning chunk boundaries

**Example:**

```
Chunk 1: [0-256 tokens]
Chunk 2: [206-462 tokens]  ← 50 token overlap
Chunk 3: [412-668 tokens]  ← 50 token overlap
```

### Hierarchy Depth

**Chosen Depth: 3 Levels (Small → Medium → Large)**

**Rationale:**

-   **3 levels** provide sufficient granularity without excessive complexity
-   Covers most query types effectively:
    -   Small: Precise facts
    -   Medium: Balanced queries
    -   Large: Broad context
-   More levels would increase storage and complexity without significant benefit
-   Matches common patterns in RAG systems

**Alternative Considered: 4 Levels**

-   Would add "tiny" (128 tokens) and "extra-large" (2048 tokens)
-   Rejected due to:
    -   Diminishing returns on precision
    -   Increased storage overhead
    -   More complex merge logic

---

## Recall Improvement

### 1. Hierarchical Chunking

**How it improves recall:**

-   **Multiple granularities** ensure information is captured at appropriate levels
-   If a fact is missed at small chunk level, it's likely captured at medium/large level
-   **Parent-child relationships** allow upward traversal when needed

### 2. Overlap Strategy

**How it improves recall:**

-   **Boundary coverage**: Prevents information loss at chunk boundaries
-   **Context preservation**: Related concepts aren't split across chunks
-   **Merge safety**: When merging chunks, overlap ensures continuity

### 3. Summary Index

**How it improves recall:**

-   **High-level coverage**: Summaries capture key information even if specific chunks are missed
-   **Timeline reconstruction**: Can answer queries even without exact chunk matches
-   **Fallback mechanism**: If chunk retrieval fails, summary retrieval can still answer

### 4. Metadata Filtering

**How it improves recall:**

-   **Precision targeting**: Filtering by claim_id, document_id, section_id reduces noise
-   **Focused search**: Narrower search space improves signal-to-noise ratio
-   **Structured navigation**: Can directly target relevant sections

### 5. Auto-Merging Retriever

**How it improves recall:**

-   **Adaptive context**: Automatically expands context when initial retrieval is insufficient
-   **Confidence-based merging**: Only merges when needed, avoiding unnecessary noise
-   **Hierarchical traversal**: Can access information at multiple levels

### 6. Multi-Index Strategy

**How it improves recall:**

-   **Complementary coverage**: Summary Index and Hierarchical Index cover different query types
-   **Router selection**: Ensures queries use the most appropriate index
-   **Fallback options**: If one index fails, can try the other

---

## Project Structure

```
agentic-rag/
├── README.md
├── requirements.txt
├── config/
│   └── config.yaml
├── src/
│   ├── data_ingestion/
│   │   ├── pdf_processor.py
│   │   ├── structure_identifier.py
│   │   ├── chunker.py
│   │   └── summarizer.py
│   ├── indexing/
│   │   ├── summary_index.py
│   │   ├── hierarchical_index.py
│   │   └── vector_store.py
│   ├── agents/
│   │   ├── router_agent.py
│   │   ├── summarization_agent.py
│   │   └── needle_haystack_agent.py
│   └── utils/
│       ├── llm_utils.py
│       └── evaluation.py
├── data/
│   ├── raw/
│   └── processed/
├── tests/
│   └── test_queries.json
└── main.py
```

---

## Next Steps

1. Set up Python environment and install dependencies
2. Implement PDF processing and structure identification
3. Build hierarchical chunking pipeline
4. Create Summary Index with MapReduce
5. Implement vector store (ChromaDB) integration
6. Build Router Agent with query classification
7. Implement Summarization Expert Agent
8. Implement Needle-in-a-Haystack Agent with Auto-Merging
9. Create evaluation framework (LLM-as-a-Judge)
10. Test end-to-end system with sample queries

---

## Key Design Principles

1. **Separation of Concerns**: Clear boundaries between indexing and querying
2. **Hierarchical Thinking**: Preserve document structure throughout pipeline
3. **Adaptive Retrieval**: Use Auto-Merging to adapt to query needs
4. **Metadata-Driven**: Leverage metadata for precision and filtering
5. **Multi-Index Strategy**: Different indexes for different query types
6. **Agent Specialization**: Each agent optimized for its query type

---

## Why LangChain for Agents?

### Key Advantages for This Assignment

1. **MCP Tool Integration** (Assignment Requirement)

    - LangChain has native support for MCP (Model Context Protocol) tools
    - Easy integration with external tools and APIs
    - Better tool discovery and management

2. **Sophisticated Agent Patterns**

    - `AgentExecutor` provides robust agent execution with error handling
    - Built-in support for ReAct (Reasoning + Acting) pattern
    - Plan-and-Execute agents for complex multi-step tasks
    - Better control over agent reasoning loops

3. **Tool Calling & Function Calling**

    - Superior structured outputs using OpenAI function calling
    - Easy creation of custom tools as Python functions
    - Automatic tool selection and parameter extraction
    - Better error handling when tools fail

4. **Multi-Agent Orchestration**

    - Clean patterns for agent-to-agent communication
    - Supervisor pattern for routing between agents
    - Better state management across agent interactions

5. **Memory & Context Management**

    - Built-in conversation memory
    - Context window management
    - Better handling of long conversations

6. **Production-Ready Features**
    - Streaming responses
    - Callback handlers for monitoring
    - Better logging and debugging
    - Retry mechanisms and error recovery

### Why Not Pure LlamaIndex for Agents?

While LlamaIndex has agents (`ReActAgent`, `OpenAIAgent`), they are:

-   Less mature than LangChain's agent framework
-   More limited in tool integration
-   Less flexible for complex routing logic
-   Weaker MCP support

### Why Hybrid Approach Works Best

-   **LlamaIndex**: Excellent for RAG-specific tasks (indexing, retrieval, summarization)
-   **LangChain**: Excellent for agent orchestration, tool integration, and complex workflows
-   **Best of Both**: Use each framework for what it does best

---

## References

-   **LlamaIndex Documentation**: https://docs.llamaindex.ai/
-   **LangChain Documentation**: https://python.langchain.com/
-   **LangChain Agents Guide**: https://python.langchain.com/docs/modules/agents/
-   **ChromaDB Documentation**: https://docs.trychroma.com/
-   **Auto-Merging Retriever Pattern**: LlamaIndex Advanced Retrieval Guide
-   **MapReduce Summarization**: LlamaIndex Summary Index Guide
-   **MCP (Model Context Protocol)**: https://modelcontextprotocol.io/
