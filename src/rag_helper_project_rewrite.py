from multiprocessing import context
import os
import json
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from google import genai
from google.genai import types

from index_rewrite import hybrid_search
from src.metrics import LLMCallRecord, calculate_cost

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set")

client = genai.Client(api_key=GEMINI_API_KEY)

INSTRUCTIONS = """
You answer questions about earnings call transcripts.

Use only the provided transcript context.
Do not use outside knowledge or make unsupported inferences.

Give a direct, concise answer to the question.
Include the key figures, time periods, and drivers when they are stated
in the context.

If the context does not contain enough information to answer fully,
say: "I don't know based on the provided transcript context."

Do not mention the retrieval process or say "the context says."
""".strip()


PROMPT_TEMPLATE = """
Question:
{question}

Context:
{context}
""".strip()

# Use an LLM for entity extraction, ticker matching, temporal extraction, and query rewriting
QUERY_REWRITE_PROMPT = """
You optimize retrieval requests for a financial earnings-call transcript RAG system.

Return valid JSON only:


{{
  "search_query": "concise keyword-rich search query",
  "symbols": ["TICKER"],
  "year": null,
  "quarter": null
}}

Rules:
- Identify public companies mentioned in the question, including likely
  spelling mistakes, abbreviations, common aliases, and former names.
- For each company, return its US-listed ticker in symbols when highly confident.
- Return an empty symbols array if the company is ambiguous or uncertain.
- Include the company name and ticker in search_query when a ticker is found.
- Extract only explicitly stated fiscal years and quarters.
- Use quarter as an integer from 1 to 4, otherwise null.
- Include relevant financial concepts: revenue, margins, guidance, demand,
  pricing, capex, inventory, AI, buybacks, or free cash flow.
- Do not invent companies, tickers, dates, quarters, metrics, or events.
- Keep search_query between 5 and 20 words.
- Return JSON only. Do not use Markdown fences or explanatory text.

User question:
{question}
""".strip()


QUERY_REWRITE_SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "search_query": {
            "type": "STRING",
        },
        "symbols": {
            "type": "ARRAY",
            "items": {
                "type": "STRING",
            },
        },
        "year": {
            "type": "INTEGER",
            "nullable": True,
        },
        "quarter": {
            "type": "INTEGER",
            "nullable": True,
        },
    },
    "required": [
        "search_query",
        "symbols",
        "year",
        "quarter",
    ],
}



class TranscriptRAG:
    def __init__(
        self,
        client,
        model: str = "gemini-3.6-flash",
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
    ):
        """Initialize the RAG helper with an LLM client, model, and prompt settings."""
        self.client = client
        self.model = model
        self.instructions = instructions
        self.prompt_template = prompt_template
        self.last_call = None

    def rewrite_query(self, question: str) -> Dict[str, Any]:
        """
        Use Gemini to resolve a company/ticker, extract time filters,
        and create a concise query for hybrid retrieval.
        """
        fallback = {
            "search_query": question,
            "symbols": [],
            "year": None,
            "quarter": None,
        }

        try:
            response = self.client.models.generate_content(
                model=self.model,
                contents=QUERY_REWRITE_PROMPT.format(question=question),
                config=types.GenerateContentConfig(
                    temperature=0.0,
                    response_mime_type="application/json",
                    response_schema=QUERY_REWRITE_SCHEMA,
                    max_output_tokens=1024,
                ),
            )

            # data = json.loads(response.text)
            # Extract text from all text parts in the response
            text_parts = []
            if response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text_parts.append(part.text)

            raw_response = "".join(text_parts).strip()

            print("\n=== RAW QUERY REWRITE RESPONSE ===")
            print(repr(raw_response))

            if not raw_response:
                print(f"Finish reason: {response.candidates[0].finish_reason}")
                raise ValueError("Gemini returned an empty query-rewrite response")

            data = json.loads(raw_response)

            search_query = str(data.get("search_query") or question).strip()

            symbols = [
                str(symbol).upper().strip()
                for symbol in (data.get("symbols") or [])
                if str(symbol).strip()
            ]

            year = data.get("year")
            year = int(year) if year is not None else None

            quarter = data.get("quarter")
            quarter = int(quarter) if quarter is not None else None
            if quarter not in {None, 1, 2, 3, 4}:
                quarter = None

            return {
                "search_query": search_query,
                "symbols": list(dict.fromkeys(symbols)),
                "year": year,
                "quarter": quarter,
            }

        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            print(f"Query rewriting failed: {exc}. Using original query.")
            return fallback
        except Exception as exc:
            print(f"Query rewriting failed: {exc}. Using original query.")
            return fallback

    
    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
        """Rewrite, extract filters, and retrieve transcript chunks."""
        retrieval = self.rewrite_query(query)

        print("\n=== QUERY REWRITE ===")
        print(f"Original query:   {query}")
        print(f"Search query:     {retrieval['search_query']}")
        print(f"Symbols:          {retrieval['symbols']}")
        print(f"Year:             {retrieval['year']}")
        print(f"Quarter:          {retrieval['quarter']}")

        # Apply a strict company filter only for one identified company.
        symbol = retrieval["symbols"][0] if len(retrieval["symbols"]) == 1 else None

        results = hybrid_search(
            query=retrieval["search_query"],
            limit=limit,
            symbol=symbol,
            year=retrieval["year"],
            quarter=retrieval["quarter"],
        )

        return retrieval, results

    def build_context(self, search_results: List[Dict[str, Any]]) -> str:
        """Format retrieved transcript chunks into a single context string for the prompt."""
        parts = []
        for i, doc in enumerate(search_results, start=1):
            meta = []
            if doc.get("symbol"):
                meta.append(str(doc["symbol"]))
            if doc.get("year"):
                meta.append(str(doc["year"]))
            if doc.get("quarter"):
                meta.append(f"Q{doc['quarter']}")
            meta_str = " ".join(meta)

            parts.append(f"[{i}] {meta_str}".strip())
            parts.append(doc.get("text", ""))
            parts.append("")

        return "\n".join(parts).strip()

    def build_prompt(self, query: str, search_results: List[Dict[str, Any]]) -> str:
        """Combine the user query and retrieved context into the final LLM prompt."""
        context = self.build_context(search_results)
        return self.prompt_template.format(question=query, context=context)

    def llm(self, prompt: str) -> str:
        """Send the prompt to Gemini, record usage metrics, and return text."""
        start_time = time.time()

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
                temperature=0.2,
            ),
        )

        response_time = time.time() - start_time
        usage = response.usage_metadata

        self.last_call = LLMCallRecord(
            model=self.model,
            prompt=prompt,
            instructions=self.instructions,
            answer=response.text,
            prompt_tokens=usage.prompt_token_count or 0,
            completion_tokens=usage.candidates_token_count or 0,
            total_tokens=usage.total_token_count or 0,
            response_time=response_time,
            cost=calculate_cost(self.model, usage),
        )

        return response.text

    def rag(self, query: str, limit: int = 5) -> Dict[str, Any]:
        retrieval, search_results = self.search(query, limit=limit)

        context = self.build_context(search_results)

        print("\n=== RETRIEVED CONTEXT ===")
        print(context[:1000])  # First 1000 characters
        print("\n=== RETRIEVED CONTEXT END===")

        # Answer against the user's original wording, not the rewritten query.
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)

        return {
            "question": query,
            "retrieval_query": retrieval,
            "answer": answer,
            "sources": search_results,
            "prompt": prompt,
        }
            


ragger = TranscriptRAG(client=client)


def ask(query: str, limit: int = 5) -> Dict[str, Any]:
    """Convenience wrapper for asking a question with the default RAG instance."""
    return ragger.rag(query, limit=limit)


if __name__ == "__main__":
    q = os.getenv("TEST_QUERY", "What did Apple say about margins?")
    result = ask(q)
    print(result["answer"])
