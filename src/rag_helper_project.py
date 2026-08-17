import os
from typing import List, Dict, Any

from dotenv import load_dotenv
from google import genai
from google.genai import types

from index import hybrid_search

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


class TranscriptRAG:
    def __init__(
        self,
        client,
        model: str = "gemini-2.5-flash",
        instructions: str = INSTRUCTIONS,
        prompt_template: str = PROMPT_TEMPLATE,
    ):
        """Initialize the RAG helper with an LLM client, model, and prompt settings."""
        self.client = client
        self.model = model
        self.instructions = instructions
        self.prompt_template = prompt_template

    def search(self, query: str, limit: int = 5) -> List[Dict[str, Any]]:
        """Retrieve the top transcript chunks using hybrid text + vector search."""
        return hybrid_search(query, limit=limit)

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
        """Send the prompt to Gemini and return the generated answer text."""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=self.instructions,
                temperature=0.2,
            ),
        )
        return response.text

    def rag(self, query: str, limit: int = 5) -> Dict[str, Any]:
        """Run the full RAG pipeline: search, prompt construction, and answer generation."""
        search_results = self.search(query, limit=limit)
        prompt = self.build_prompt(query, search_results)
        answer = self.llm(prompt)
        return {
            "question": query,
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
