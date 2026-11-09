import numpy as np
import pandas as pd
import json
import sqlite3
from datetime import datetime, timezone
import csv
import time
import genAI
import mock_llm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import uuid
from pathlib import Path

file = open("data/policies/delivery_policy.txt","r")
delivery_policy=file.read()
file.close()

file = open("data/policies/escalation_policy.txt","r")
escalation_policy=file.read()
file.close()

file = open("data/policies/payment_policy.txt","r")
payment_policy=file.read()
file.close()

file = open("data/policies/privacy_policy.txt","r")
privacy_policy=file.read()
file.close()

file = open("data/policies/refund_policy.txt","r")
refund_policy=file.read()
file.close()

documents=[delivery_policy, escalation_policy, payment_policy, privacy_policy, refund_policy]


system_prompt="""Refund rules the agent's tools must implement. 
Only completed orders can be refunded; a completed order delivered in more than 45 minutes is refunded 100 percent of its order_amount; delivered in 45 minutes or less → refund 0; a refund above INR 1000 is escalated to a human and not issued automatically. 
All simulated requests are treated as raised inside the policy's 7-day window."""


CHUNK_SIZE = 40
OVERLAP = 10
STEP = CHUNK_SIZE - OVERLAP   # 30

_vectorizer = None
_matrix = None
_chunks = None

doc = ["delivery_policy", "escalation_policy", "payment_policy", "privacy_policy", "refund_policy"]

POLICY_SYSTEM_PROMPT = (
    "You are a POLICY ASSISTANT for customer support. "
    "Answer the question using only the supplied context. "
    "If the context does not contain the answer, say that you could not find the information."
)

EVAL_PATH = "data/raw/policy_eval.csv"
FALLBACK_PREFIX = "I could not find"

UNANSWERABLE_MARKERS = {"none", "not_covered", "", "nan"}

AGENT_SYSTEM_PROMPT = (
    "You are the AGENT for Swiggy Instamart customer support. Use the tools to "
    "check the order, the policy and the refund rules. Reply with a final answer "
    "in the form 'STATUS: explanation', where STATUS is REFUND, ESCALATE, DENIED or NOT_FOUND."
)


def chunk_words(words, source):
    """Split one document's words into 40-word chunks with 10-word overlap."""
    if not words:
        return []
    chunks = []
    start = 0
    while True:
        end = start + CHUNK_SIZE                     # exclusive end index
        chunks.append({
            "source": source,
            "start_word": start,
            "text": " ".join(words[start:end]),
        })
        if end >= len(words):                        # this chunk reaches or passes the last word
            break
        start += STEP
    return chunks


def chunk_all_documents(documents, names):
    """Chunk each document, tagging every chunk with its name from the names list."""
    if len(documents) != len(names):
        raise ValueError(f"{len(documents)} documents but {len(names)} names")
    all_chunks = []
    for text, name in zip(documents, names):
        all_chunks.extend(chunk_words(text.split(), name))
    return all_chunks


def fit_index(chunks):
    """Fit TF-IDF on the chunk texts and store the index for retrieve()."""
    global _vectorizer, _matrix, _chunks
    _chunks = chunks
    _vectorizer = TfidfVectorizer()
    _matrix = _vectorizer.fit_transform([c["text"] for c in chunks])


def retrieve(query, k=3):
    """Return the top-k chunks for a query, ranked by cosine similarity."""
    if _vectorizer is None:
        raise RuntimeError("Call fit_index(chunks) first")
    q = _vectorizer.transform([query])                 # same fitted vocabulary
    sims = cosine_similarity(q, _matrix)[0]            # one score per chunk
    top = np.argsort(-sims, kind="stable")[:k]         # highest score first
    return [
        {"source": _chunks[i]["source"], "score": float(sims[i]), "text": _chunks[i]["text"]}
        for i in top
    ]


def build_context(results):
    """Format retrieved chunks as [source] lines followed by the chunk text."""
    parts = [f"[{r['source']}]\n{r['text']}" for r in results]
    return "\n".join(parts)


def answer(question, k=3):
    """Retrieve k chunks, ask the policy assistant, and return the answer with its sources."""
    results = retrieve(question, k=k)
    sources = [r["source"] for r in results]

    user_content = f"CONTEXT:\n{build_context(results)}\n\nQUESTION:\n{question}"

    messages = [
        {"role": "system", "content": POLICY_SYSTEM_PROMPT},   # contains POLICY ASSISTANT
        {"role": "user", "content": user_content},
    ]
    resp = mock_llm.chat(messages)

    return {"answer": genAI.get_reply_text(resp), "sources": sources}

def chunk_all_documents_with_overlap(documents, names, overlap):
    """Rebuild chunks with the given overlap (step = 40 - overlap)."""
    global CHUNK_SIZE, OVERLAP, STEP
    CHUNK_SIZE, OVERLAP, STEP = 40, overlap, 40 - overlap
    return chunk_all_documents(documents, names)


def clean_doc(value):
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in UNANSWERABLE_MARKERS:
        return None
    return v.replace(".txt", "").replace(" ", "_")   # match names like refund_policy

def evaluate(eval_df, chunks):
    fit_index(chunks)
    rows = []
    for _, row in eval_df.iterrows():
        q = row["question"]
        expected_doc = clean_doc(row["expected_doc"])
        expected_fact = row["expected_fact"] if isinstance(row["expected_fact"], str) and expected_doc else None

        retrieved = retrieve(q, k=3)
        top_sources = [r["source"] for r in retrieved]
        result = answer(q)

        rows.append({
            "question": q,
            "answerable": expected_doc is not None,
            "doc_in_top3": (expected_doc in top_sources) if expected_doc else None,
            "fact_in_answer": (expected_fact in result["answer"]) if expected_fact else None,
            "fallback": result["answer"].startswith(FALLBACK_PREFIX),
            "answer": result["answer"],
        })
    return pd.DataFrame(rows)


def report(df, label):
    ans = df[df["answerable"]]
    unans = df[~df["answerable"]]
    print(f"\n\n--- {label} ---")
    print(f"Answerable: expected_doc in top 3 = {int(ans['doc_in_top3'].sum())}/{len(ans)}")
    print(f"Answerable: expected_fact in answer = {int(ans['fact_in_answer'].sum())}/{len(ans)}")
    print(f"Unanswerable: fallback given = {int(unans['fallback'].sum())}/{len(unans)}")
    print()


    

def _append_ledger(tool_name, order_id, refund_inr):
    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "tool": tool_name,
        "order_id": order_id,
        "refund_inr": refund_inr,
    }
    with open("outputs/refund_ledger.jsonl", "a", encoding="utf-8") as file:
        file.write(json.dumps(entry) + "\n")


def get_order(order_id):
    conn = sqlite3.connect("data/instamart.db")
    conn.row_factory = sqlite3.Row  # lets us turn the row into a dict
    row = conn.execute(
        "SELECT order_id, status, order_amount, delivery_minutes "
        "FROM orders WHERE order_id = ?",
        (order_id,),
    ).fetchone()
    conn.close()

    if row is None:
        return {"error": "order not found"}
    return dict(row)


def search_policy(query):
    results = retrieve(query, k=3)  # same retriever that answer() uses
    return {"results": results}


def calculate_refund(order_amount, delivery_minutes):
    if delivery_minutes > 45:
        refund_inr = float(order_amount)  # late delivery: full refund
    else:
        refund_inr = 0.0                  # on time (45 min or less): no refund
    return {"refund_inr": refund_inr}


def issue_refund(order_id, refund_inr):
    _append_ledger("issue_refund", order_id, refund_inr)
    return {"refund_inr": refund_inr}


def escalate_to_human(order_id, refund_inr):
    _append_ledger("escalate_to_human", order_id, refund_inr)
    return {"refund_inr": refund_inr}


TOOLS_SPEC = [
    {
        "name": "get_order",
        "description": "Fetch the status, amount and delivery time of an order.",
        "parameters": {
            "type": "object",
            "properties": {"order_id": {"type": "string"}},
            "required": ["order_id"],
        },
    },
    {
        "name": "search_policy",
        "description": "Retrieve company policy passages relevant to a customer query.",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    {
        "name": "calculate_refund",
        "description": "Compute the refund amount in INR from order amount and delivery time.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_amount": {"type": "number"},
                "delivery_minutes": {"type": "number"},
            },
            "required": ["order_amount", "delivery_minutes"],
        },
    },
    {
        "name": "issue_refund",
        "description": "Issue a refund for an order and record it in the ledger.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
    {
        "name": "escalate_to_human",
        "description": "Escalate an order's refund case to a human agent and log it.",
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {"type": "string"},
                "refund_inr": {"type": "number"},
            },
            "required": ["order_id", "refund_inr"],
        },
    },
]

TOOL_MAP = {
    "get_order": get_order,
    "search_policy": search_policy,
    "calculate_refund": calculate_refund,
    "issue_refund": issue_refund,
    "escalate_to_human": escalate_to_human,
}


def run_tool_call(name, arguments):
    func = TOOL_MAP.get(name)
    if func is None:
        return {"error": f"unknown tool: {name}"}
    return func(**arguments)





def _estimate_tokens(text):
    # rough estimate: about 4 characters per token
    return len(text) // 4

STATE_DIR = Path("state")


class SimulatedCrash(Exception):
    """Raised on purpose to simulate the process dying mid-run."""
    pass


def _save_checkpoint(state):
    # everything needed to continue from the next model call
    STATE_DIR.mkdir(exist_ok=True)
    data = {
        "run_id": state["run_id"],
        "scenario_id": state["scenario_id"],
        "messages": state["messages"],
        "tokens_used": state["tokens_used"],
        "llm_calls": state["llm_calls"],
        "step_number": len(state["steps"]),
        "tools_executed": state["tools_executed"],
        "steps": state["steps"],
    }
    with open(STATE_DIR / f"{state['run_id']}.json", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _agent_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool=None):
    status = None
    final_answer = None
    start = time.monotonic()  # the clock restarts on resume

    # on resume, only the remaining model calls are allowed
    for _ in range(max_iterations - state["llm_calls"]):
        # (c) wall-clock limit
        if time.monotonic() - start >= time_limit_seconds:
            status = "FALLBACK_TIMEOUT"
            break

        resp = mock_llm.chat(state["messages"], tools=TOOLS_SPEC)
        state["llm_calls"] += 1

        # (b) token budget, checked before any requested tool runs
        state["tokens_used"] += resp.usage.total_tokens
        if state["tokens_used"] > token_limit:
            status = "FALLBACK_TOKEN_LIMIT"
            break

        msg = resp.choices[0].message
        reply_text = msg.content or ""
        tool_calls = msg.tool_calls or []

        if not tool_calls:
            final_answer = reply_text
            status = final_answer.split(":")[0].strip()
            break

        parsed_calls = []
        for tc in tool_calls:
            args = tc.arguments
            if isinstance(args, str):
                args = json.loads(args)
            parsed_calls.append({"id": tc.id, "name": tc.name, "arguments": args})

        state["messages"].append({
            "role": "assistant",
            "content": None,
            "tool_calls": parsed_calls,
        })

        for tc in parsed_calls:
            result = run_tool_call(tc["name"], tc["arguments"])
            state["steps"].append({
                "tool": tc["name"],
                "arguments": tc["arguments"],
                "observation": result,
            })
            state["tools_executed"].append(tc["name"])
            state["messages"].append({
                "role": "tool",
                "tool_call_id": tc["id"],
                "name": tc["name"],
                "content": json.dumps(result),
            })

            # checkpoint after every observation
            _save_checkpoint(state)

            # simulated crash, right after this tool's checkpoint is saved
            if crash_after_tool is not None and len(state["tools_executed"]) == crash_after_tool:
                raise SimulatedCrash(f"crash after tool #{crash_after_tool} ({tc['name']})")

    # (a) the loop ran out without a final answer or a stop condition
    if status is None:
        status = "FALLBACK_MAX_ITERATIONS"

    if status.startswith("FALLBACK"):
        final_answer = _handoff_summary(status, state["steps"], state["llm_calls"])

    return {
        "scenario_id": state["scenario_id"],
        "status": status,
        "final_answer": final_answer,
        "steps": state["steps"],
        "llm_calls": state["llm_calls"],
        "tokens_used": state["tokens_used"],
    }


def run_agent(message, scenario_id, max_iterations=6, token_limit=6000,
              time_limit_seconds=10, run_id=None, crash_after_tool=None):
    state = {
        "run_id": run_id or uuid.uuid4().hex[:8],
        "scenario_id": scenario_id,
        "messages": [
            {"role": "system", "content": AGENT_SYSTEM_PROMPT},
            {"role": "user", "content": message},
        ],
        "tokens_used": 0,
        "llm_calls": 0,
        "steps": [],
        "tools_executed": [],
    }
    return _agent_loop(state, max_iterations, token_limit, time_limit_seconds, crash_after_tool)


def resume_agent(run_id, max_iterations=6, token_limit=6000, time_limit_seconds=10):
    with open(STATE_DIR / f"{run_id}.json", encoding="utf-8") as f:
        data = json.load(f)
    state = {
        "run_id": data["run_id"],
        "scenario_id": data["scenario_id"],
        "messages": data["messages"],
        "tokens_used": data["tokens_used"],
        "llm_calls": data["llm_calls"],
        "steps": data["steps"],
        "tools_executed": data["tools_executed"],
    }
    return _agent_loop(state, max_iterations, token_limit, time_limit_seconds)

def _handoff_summary(status, steps, llm_calls):
    tools_used = ", ".join(s["tool"] for s in steps) or "none"
    return (
        f"HANDOFF: agent stopped with {status} after {llm_calls} model calls. "
        f"Tools run: {tools_used}. Please review this case and continue manually."
    )

def get_refund_amount(steps):
    amount = None
    for step in steps:
        obs = step.get("observation", {})
        if isinstance(obs, dict) and "refund_inr" in obs:
            amount = obs["refund_inr"]
    return amount

def main():
    open("outputs/refund_ledger.jsonl", "w").close()  # clear the ledger for a fresh simulation run
    rows = list(csv.DictReader(open("data/raw/agent_scenarios.csv", newline="", encoding="utf-8")))

    with open("outputs/agent_runs.jsonl", "w", encoding="utf-8") as out:
        for row in rows:
            result = run_agent(row["message"], row["scenario_id"])
            record = {
                "scenario_id": row["scenario_id"],
                "status": result["status"],
                "final_answer": result["final_answer"],
                "refund_inr": get_refund_amount(result["steps"]),
                "llm_calls": result["llm_calls"],
                "tokens_used": result["tokens_used"],
                "steps": result["steps"],
            }
            out.write(json.dumps(record) + "\n")
            print(record["scenario_id"], record["status"], record["refund_inr"],
                  record["llm_calls"], record["tokens_used"])




if __name__ == "__main__":
    chunks = chunk_all_documents(documents, doc)

    for i, c in enumerate(chunks):
        print(f"Chunk {i} [{c['source']}, words {c['start_word']}-{c['start_word'] + CHUNK_SIZE - 1}]: {c['text'][:80]}...")

    print("\nTotal chunks:", len(chunks))
    counts = {}
    for c in chunks:
        counts[c["source"]] = counts.get(c["source"], 0) + 1
    print("Chunks per document:", counts)

    chunks = chunk_all_documents(documents, doc)
    fit_index(chunks)

    results = retrieve("refund policy late delivery", k=3)
    for r in results:
        print(f"\n\n{r['source']}  score={r['score']:.4f}")
        print(f"  {r['text'][:100]}...")

    result = answer("Can I get a refund if my delivery is late?")
    print("\n\nAnswer:", result["answer"])
    print("Sources:", result["sources"])

    eval_df = pd.read_csv(EVAL_PATH)
    names = doc
    docs = documents

    # Run 1: 40-word chunks, overlap 10
    chunks_10 = chunk_all_documents_with_overlap(docs, names, overlap=10)
    res_10 = evaluate(eval_df, chunks_10)
    report(res_10, "overlap 10")

    # Run 2: 40-word chunks, overlap 0
    chunks_0 = chunk_all_documents_with_overlap(docs, names, overlap=0)
    res_0 = evaluate(eval_df, chunks_0)
    report(res_0, "overlap 0")

    # Questions that fail with overlap 0 but pass with overlap 10
    broken = res_0[res_0["answerable"] & (res_0["fact_in_answer"] == False)]
    print("\n\nAnswerable questions failing with overlap 0:")
    print(broken[["question", "doc_in_top3", "fact_in_answer"]].to_string(index=False))


    print("\nfor order id ORD1001:",run_tool_call("get_order", {"order_id": "ORD1001"}))


    main()    

    rows = list(csv.DictReader(open("data/raw/agent_scenarios.csv", newline="", encoding="utf-8")))
    s1 = next(r for r in rows if r["scenario_id"] == "S1")

    demo_tokens = run_agent(s1["message"], "S1", token_limit=200)
    print("\n\n\nToken demo:", demo_tokens["status"], demo_tokens["llm_calls"], demo_tokens["tokens_used"])
    print(demo_tokens["final_answer"])

    demo_time = run_agent(s1["message"], "S1", time_limit_seconds=0)
    print("\n\nTimeout demo:", demo_time["status"], demo_time["llm_calls"], demo_time["tokens_used"])
    print(demo_time["final_answer"])


    rows = list(csv.DictReader(open("data/raw/agent_scenarios.csv", newline="", encoding="utf-8")))
    s1 = next(r for r in rows if r["scenario_id"] == "S1")

    try:
        run_agent(s1["message"], "S1", run_id="S1_crash_demo", crash_after_tool=2)
    except SimulatedCrash as e:
        print("\n\nCrashed:", e)

    result = resume_agent("S1_crash_demo")
    print("\nStatus:", result["status"])
    print("Refund:", get_refund_amount(result["steps"]))
    print("\nTools executed, in order:")
    for i, step in enumerate(result["steps"], 1):
        print(i, step["tool"])