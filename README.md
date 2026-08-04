# Earnings Call RAG Assistant

A Retrieval-Augmented Generation (RAG) application for querying earnings call transcripts with natural language. The system helps users quickly find relevant management commentary on topics such as revenue, margins, guidance, risks, and strategy without manually reading long transcripts.

## Problem Description

Earnings calls are one of the most important sources of information for understanding how a public company is performing and what management expects going forward. They contain detailed discussion of financial results, operating trends, risks, and future plans, but the content is usually spread across long, dense, and unstructured transcripts.

This project addresses that problem by building a RAG assistant that indexes earnings call transcripts and allows users to ask questions in plain English. Instead of searching through full transcripts by hand, users can retrieve the most relevant context and receive a concise answer grounded in the transcript text.

The application is especially useful for investors, analysts, and researchers who want a faster and more interactive way to explore earnings-call content while still keeping the answer traceable to the original source.

## Project Goal

The goal of this project is to create an interactive assistant that:

- ingests earnings call transcripts,
- chunks and indexes the text for retrieval,
- uses hybrid search to find relevant context,
- generates grounded answers with an LLM,
- stores query and feedback data for monitoring and evaluation,
- and presents the results through a Streamlit interface.

## Project Structure

```text
src/
├── app.py                 # Main Streamlit app
├── dashboard.py           # Monitoring dashboard
├── rag_helper_project.py  # RAG wrapper used by the app
├── ingestion/             # Transcript ingestion and indexing logic
├── db_init.py             # Database setup
├── db_save.py             # Save conversations to Postgres
├── db_query.py            # Query stored conversations and stats
├── metrics.py             # Tracks latency, tokens, and cost
├── evaluation_utils.py    # Utilities for evaluation experiments
└── data/                  # Raw / processed transcript data
```

> Note: this structure may be simplified or adjusted as the project is finalized.

## Retrieval Flow

1. The user enters a question in the Streamlit app.
2. The app sends the query to the RAG helper.
3. Hybrid search retrieves the most relevant transcript chunks.
4. The retrieved chunks are formatted into a prompt context.
5. Gemini generates a grounded answer using only the provided context.
6. The answer and supporting sources are displayed back to the user.
7. The interaction can be saved for later analysis and monitoring.

### High-level architecture

```text
User question → Streamlit UI → Hybrid retrieval → Prompt assembly → Gemini LLM → Answer + sources
```

## Interface

The application provides a Streamlit-based user interface with:

- a question input box,
- an Ask button to generate answers,
- a Clear button to reset the input,
- a sidebar for settings such as top-k chunks,
- and example queries to help users get started.

The interface is intentionally simple so users can focus on asking questions and reviewing grounded answers from earnings call transcripts.

## Monitoring

The project includes monitoring and logging features so usage can be tracked over time. Conversation records store fields such as:

- question,
- answer,
- model,
- prompt,
- token usage,
- response time,
- estimated cost,
- and timestamp.

A dashboard can display summary metrics such as:

- total conversations,
- average response time,
- total cost,
- average tokens,
- cost over time,
- response time over time,
- and recent conversations.

These metrics make it easier to evaluate system behavior and identify opportunities for improvement.

## Data Pipeline

The data pipeline is responsible for preparing earnings call transcripts for retrieval. In general, it includes:

- collecting transcript data,
- cleaning and normalizing the text,
- chunking transcripts into manageable passages,
- generating embeddings,
- storing searchable records in the database or vector index,
- and making the content available to the retrieval layer.

## Evaluation

The project can be evaluated on both retrieval quality and answer quality. Useful criteria include:

- whether the top retrieved chunks are relevant,
- whether the answer is grounded in the transcript,
- whether the system avoids hallucinations,
- and whether the response is concise and useful.

Additional evaluation can include comparing different retrieval settings, different chunk sizes, and different prompting strategies.

## Reproducibility

The project is designed to be reproducible with clear scripts and environment setup. A complete version of the repository should include:

- installation instructions,
- environment variables,
- database setup steps,
- ingestion or indexing commands,
- and instructions for running the Streamlit app and dashboard.

## Installation

```bash
git clone <repo-url>
cd <repo-folder>
pip install -r requirements.txt
```

Set the required environment variables before running the app.

## Usage

Start the main app:

```bash
streamlit run src/app.py
```

Start the dashboard if it is in a separate app:

```bash
streamlit run src/dashboard.py
```

## Limitations

This project is a draft implementation and may still have limitations such as:

- incomplete transcript coverage,
- dependence on retrieval quality,
- limited evaluation data,
- and possible formatting differences across transcript sources.

The final version should document these limitations clearly.

## Future Work

Potential next steps include:

- improving retrieval with better chunking or reranking,
- adding filters for ticker, year, and quarter,
- expanding evaluation metrics,
- improving citation display in the UI,
- and refining monitoring dashboards.

## License

To be added.

## Acknowledgments

This project is inspired by the LLM Zoomcamp style of RAG applications and by the need to make long-form financial disclosures easier to explore.
