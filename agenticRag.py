"""
Agentic RAG on top of the FiQA hybrid retrieval pipeline.

Instead of always running the same fixed retrieval call, the agent is given
tool access to the existing retrievers (search_bm25, search_dense,
search_hybrid_boost, search_reranked) and decides for itself: which
method(s) to call, how to phrase the query, and whether to search again if
the first attempt doesn't look sufficient -- rather than every query running
an identical, pre-decided pipeline.

Uses OpenRouter (OpenAI-compatible API) so any tool-calling-capable model
can drive the agent with a single OPENROUTER_API_KEY.

Requires OPENROUTER_API_KEY in .env.

Usage:
    python agentic_rag.py
"""

import json
from typing import Any, cast

from openai import OpenAI
from openai.types.chat import ChatCompletionMessageParam, ChatCompletionToolParam

from config import AGENT_MODEL, OPENROUTER_API_KEY, OPENROUTER_BASE_URL
from rerank import search_reranked, search_reranked_boost
from retrievers import search_bm25, search_dense, search_hybrid_boost, search_hybrid_rrf

if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY environment variable is not set")

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url=OPENROUTER_BASE_URL)

MAX_TOOL_ROUNDS = 5  # hard cap so a confused agent can't loop forever


# ---------------------------------------------------------------------------
# Tool definitions -- thin wrappers around the existing retrievers, each
# returning JSON-serializable results the model can read and reason about.
# ---------------------------------------------------------------------------
def _format_results(results: list[tuple[str, float, str]]) -> list[dict]:
    return [
        {"doc_id": doc_id, "score": round(score, 4), "text": text[:500]}
        for doc_id, score, text in results
    ]


TOOLS = cast(
    list[ChatCompletionToolParam],
    [
        {
            "type": "function",
            "function": {
                "name": "search_keyword",
                "description": "Keyword (BM25) search. Best for exact terms, ticker symbols, tax form codes, regulation names -- anything where exact wording matters.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_semantic",
                "description": "Dense/semantic search. Best for paraphrase and conceptual questions where the answer may use different wording than the query.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_hybrid",
                "description": "Combined keyword + semantic search (normalized score blend). Good general-purpose default when unsure which single method fits better. Best overall standalone method on this dataset.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_hybrid_rrf",
                "description": "Combined keyword + semantic search fused by Reciprocal Rank Fusion (rank-based, ignores score magnitude). Alternative to search_hybrid -- try this if search_hybrid's results look off, since the two fusion methods can disagree.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_hybrid_reranked",
                "description": "search_hybrid_rrf followed by cross-encoder reranking. Slower than the other methods -- use when the first search's results look weak or ambiguous and you need a stronger pass.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "search_hybrid_boost_reranked",
                "description": "search_hybrid followed by cross-encoder reranking. Alternative escalation path to search_hybrid_reranked -- reranks the normalized-blend candidates instead of the RRF candidates.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query."},
                        "top_k": {
                            "type": "integer",
                            "description": "Number of results to return.",
                            "default": 5,
                        },
                    },
                    "required": ["query"],
                },
            },
        },
    ],
)

TOOL_FUNCTIONS = {
    "search_keyword": lambda query, top_k=5: _format_results(
        search_bm25(query, top_k=top_k)
    ),
    "search_semantic": lambda query, top_k=5: _format_results(
        search_dense(query, top_k=top_k)
    ),
    "search_hybrid": lambda query, top_k=5: _format_results(
        search_hybrid_boost(query, top_k=top_k)
    ),
    "search_hybrid_rrf": lambda query, top_k=5: _format_results(
        search_hybrid_rrf(query, top_k=top_k)
    ),
    "search_hybrid_reranked": lambda query, top_k=5: _format_results(
        search_reranked(query, top_k=top_k)
    ),
    "search_hybrid_boost_reranked": lambda query, top_k=5: _format_results(
        search_reranked_boost(query, top_k=top_k)
    ),
}


# ---------------------------------------------------------------------------
# Agent loop
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """You are a financial Q&A assistant with access to a FiQA
document search system via six tools: search_keyword, search_semantic,
search_hybrid, search_hybrid_rrf, search_hybrid_reranked, and
search_hybrid_boost_reranked. Choose whichever tool(s) fit the question --
you don't need to call all of them. You may search more than once (e.g.
reformulate the query, try an alternative fusion method, or escalate to a
reranked search) if the first results don't look sufficient. Once you have
enough information, answer the user's question directly, citing which
doc_id(s) you used."""


def run_agent(question: str, verbose: bool = True) -> str:
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for round_num in range(MAX_TOOL_ROUNDS):
        response = client.chat.completions.create(
            model=AGENT_MODEL,
            messages=cast(list[ChatCompletionMessageParam], messages),
            tools=TOOLS,
        )
        message = response.choices[0].message
        messages.append(message.model_dump(exclude_none=True))

        if not message.tool_calls:
            return message.content or ""

        for tool_call in message.tool_calls:
            # Only "function" tool calls have a .function attribute -- the
            # SDK also models a "custom" tool call type we don't use here,
            # so narrow the type before accessing .function.
            if tool_call.type != "function":
                continue

            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)

            if verbose:
                print(f"  [round {round_num + 1}] calling {fn_name}({fn_args})")

            fn = TOOL_FUNCTIONS.get(fn_name)
            result = fn(**fn_args) if fn else {"error": f"Unknown tool {fn_name}"}

            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": json.dumps(result),
                }
            )

    return "Reached max tool-call rounds without a final answer."


if __name__ == "__main__":
    question = "how should i start investing in the stock market? and where?"
    print(f"Question: {question}\n")
    answer = run_agent(question)
    print(f"\nAnswer:\n{answer}")
