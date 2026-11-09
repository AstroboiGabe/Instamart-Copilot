import json, re
import pandas as pd
import logging
import random
import time  
import mock_llm
from mock_llm import TransientLLMError
from collections import Counter
from types import SimpleNamespace as NS


ZERO_SHOT_PROMPT = """You are a support ticket triage assistant. CLASSIFY each customer ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud

Reply with JSON only, with no text before or after it, in this form:
{"category": "<one of the four categories>", "reason": "<one short sentence>"}
"""

FEW_SHOT_PROMPT = """You are a support ticket triage assistant. CLASSIFY each customer ticket into exactly one of these categories:
- billing
- delivery
- technical
- fraud

Reply with JSON only, with no text before or after it, in this form:
{"category": "<one of the four categories>", "reason": "<one short sentence>"}

EXAMPLES:
Ticket: "I was charged twice for the same order."
Output: {"category": "billing", "reason": "Customer reports a duplicate charge."}

Ticket: "My order arrived 40 minutes late and the rider never called."
Output: {"category": "delivery", "reason": "Customer reports a late delivery."}

Ticket: "Someone used my account to place 5 orders I did not make."
Output: {"category": "fraud", "reason": "Customer reports unauthorised orders on their account."}
"""

PII_PATTERNS = [
    ("email",    re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[EMAIL]"),
    ("phone",    re.compile(r"\b[6-9]\d{9}\b"),          "[PHONE]"),
    ("order_id", re.compile(r"ORD-\d{6}"),               "[ORDER_ID]"),
]

def redact(text):
    """Return (redacted_text, Counter of substitutions per type) for one ticket."""
    # Convert missing values (NaN) to an empty string before regex runs
    if not isinstance(text, str):
        text = "" if pd.isna(text) else str(text)

    counts = Counter()
    for name, pattern, placeholder in PII_PATTERNS:
        text, n = pattern.subn(placeholder, text)
        counts[name] += n
    return text, counts



def redact_tickets(tickets):
    """Redact every ticket once and return the cleaned list plus total counts."""
    total = Counter()
    cleaned = []
    for tid, text in tickets:
        new_text, counts = redact(text)
        total.update(counts)
        cleaned.append((tid, new_text))
    total["total"] = sum(total[k] for k in ("email", "phone", "order_id"))
    return cleaned, total


def validate_input(ticket):
    if not ticket.strip():
        return False, "Ticket is empty"
    if len(ticket) > 2000:
        return False, "Ticket is too long"
    return True, "Valid input"

def get_reply_text(resp):
    """Pull the text out of the mock's response object."""
    if isinstance(resp, str):
        return resp
    return resp.choices[0].message.content   # confirm this path with vars(resp)

def get_total_tokens(resp):
    return resp.usage.total_tokens


ALLOWED = {"billing", "delivery", "technical", "fraud"}


def parse_reply(text):
    """Return a dict with status set to classified_json, classified_keyword, or flagged_invalid_output."""
    # 1. Try valid JSON with an allowed category
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        data = None
        # 2. Not JSON: look for exactly one allowed word among the lowercase words
        words = re.findall(r"[a-z]+", text.lower())
        found = {w for w in words if w in ALLOWED}
        if len(found) == 1:
            return {"status": "classified_keyword", "category": found.pop(), "reason": None}
        return {"status": "flagged_invalid_output", "category": None, "reason": None}

    # Valid JSON: accept only if category is one of the four allowed values
    if isinstance(data, dict) and data.get("category") in ALLOWED:
        return {"status": "classified_json",
                "category": data["category"],
                "reason": data.get("reason")}

    # Valid JSON but outside the list (or not an object): never guess
    return {"status": "flagged_invalid_output", "category": None, "reason": None}

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

BASE_DELAY = 0.05
logger = logging.getLogger(__name__)


def call_classifier(text, system_prompt=ZERO_SHOT_PROMPT):
    """Send one ticket through the mock's chat() entry point."""
    messages = [
        {"role": "system", "content": system_prompt},   # must contain CLASSIFY
        {"role": "user", "content": text},
    ]
    return mock_llm.chat(messages)


def classify_with_retries(ticket_id, text, system_prompt=ZERO_SHOT_PROMPT):
    """Try once, then retry up to 3 times with exponential backoff and jitter."""
    try:
        return call_classifier(text, system_prompt)
    except TransientLLMError as first_error:
        last_error = first_error

    for n in range(3):                                   # retry numbers 0, 1, 2
        wait = BASE_DELAY * (2 ** n) * random.uniform(0.9, 1.15)
        logger.info("ticket=%s retry=%d wait=%.4fs", ticket_id, n, wait)
        time.sleep(wait)
        try:
            return call_classifier(text, system_prompt)
        except TransientLLMError as e:
            last_error = e

    raise last_error                                     # all retries failed





FAULT_PLAN = {"target": 0, "remaining": 0, "hit": set(), "injected": 0, "caught": 0}


def plan_transient_faults(tickets, target=19):
    """Choose the first `target` valid tickets; each gets one injected failure that its retry recovers from."""
    eligible = []
    for tid, text in tickets:
        if not isinstance(text, str):
            continue
        cleaned, _ = redact(text)
        ok, _ = validate_input(cleaned)
        if ok:
            eligible.append(tid)
    if len(eligible) < target:
        raise ValueError(f"Only {len(eligible)} valid tickets; need {target}")
    FAULT_PLAN.update(target=target, remaining=target, hit=set(), injected=0, caught=0)
    FAULT_PLAN["planned_ids"] = set(eligible[:target])


def call_classifier(text, system_prompt=ZERO_SHOT_PROMPT, ticket_id=None):
    """Send one redacted ticket through chat(), injecting a fault on the first attempt of planned tickets."""
    if (ticket_id is not None
            and ticket_id in FAULT_PLAN.get("planned_ids", set())
            and ticket_id not in FAULT_PLAN["hit"]):
        # Fail only the first attempt for this ticket, so its retry succeeds
        FAULT_PLAN["hit"].add(ticket_id)
        FAULT_PLAN["remaining"] -= 1
        FAULT_PLAN["injected"] += 1
        raise TransientLLMError(f"injected timeout for ticket {ticket_id}")

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": text},
    ]
    return mock_llm.chat(messages)



def run_all(tickets, system_prompt=ZERO_SHOT_PROMPT):
    """Redact, validate, then classify. Invalid tickets are skipped and reported."""
    cleaned, _ = redact_tickets(tickets)
    mock_llm.reset()

    results, rejected = {}, {}
    for tid, text in cleaned:
        ok, reason = validate_input(text)
        if not ok:
            rejected[tid] = reason
            logger.info("ticket=%s rejected reason=%s", tid, reason)
            continue
        results[tid] = classify_with_retries(tid, text, system_prompt)

    return results, rejected




def get_reply_text(resp):
    """Pull the text out of the mock's response object."""
    if isinstance(resp, str):
        return resp
    return resp.choices[0].message.content   


def get_total_tokens(resp):
    # ASSUMPTION: the brief says response.usage.total_tokens; confirm the attribute exists.
    return resp.usage.total_tokens



TOKEN_BUDGET = 6000

def run_pipeline_budgeted(system_prompt, tickets_df, token_budget=TOKEN_BUDGET):
    """Zero-shot pipeline with a running token budget. No call is made once the budget is spent."""
    mock_llm.reset()
    spent = 0
    rows = []
    for _, row in tickets_df.iterrows():
        base = {"ticket_id": row["ticket_id"], "true_category": row["true_category"]}
        cleaned_text, _ = redact(row["text"])
        ok, _ = validate_input(cleaned_text)
        if not ok:
            rows.append({**base, "status": "rejected_input", "pred_category": None, "tokens": 0})
            continue
        if spent >= token_budget:
            # Budget already spent: make no call, route to human review
            rows.append({**base, "status": "budget_exhausted", "pred_category": None, "tokens": 0})
            continue
        resp = classify_with_retries(row["ticket_id"], cleaned_text, system_prompt)
        used = get_total_tokens(resp)
        spent += used
        parsed = parse_reply(get_reply_text(resp))
        rows.append({**base, "status": parsed["status"], "pred_category": parsed["category"], "tokens": used})
    return pd.DataFrame(rows)


TRIAGE_BUDGET = 100000
HUMAN_REVIEW_STATUSES = {"skipped_empty", "flagged_long", "flagged_invalid_output", "budget_exhausted"}


def build_triage_results(ticket_df, segments_path="outputs/customer_segments.csv"):
    # 1. Load segments: one row per customer, is_premium as a real bool
    seg = pd.read_csv(segments_path)
    seg = pd.read_csv(segments_path)
    seg = seg.drop_duplicates("customer_id").set_index("customer_id")
    seg["is_premium"] = seg["segment_name"].astype(str).str.strip().eq("Premium Buyer")

    # 2. Few-shot run with the large budget
    run_df = run_pipeline_budgeted(FEW_SHOT_PROMPT, ticket_df, TRIAGE_BUDGET)

    # 3. Attach customer_id and the original text (needed to explain rejections)
    text_by_id = dict(zip(ticket_df["ticket_id"], ticket_df["text"]))
    cust_by_id = dict(zip(ticket_df["ticket_id"], ticket_df["customer_id"]))
    df = run_df.copy()
    df["customer_id"] = df["ticket_id"].map(cust_by_id)

    # 4. Split rejected_input into skipped_empty or flagged_long
    def final_status(row):
        if row["status"] != "rejected_input":
            return row["status"]
        _, reason = validate_input(redact(text_by_id[row["ticket_id"]])[0])
        return "skipped_empty" if reason == "Ticket is empty" else "flagged_long"

    df["status"] = df.apply(final_status, axis=1)

    # 5. Blank when not available
    df["category"] = df["pred_category"].fillna("")
    df["segment_name"] = df["customer_id"].map(seg["segment_name"]).fillna("")
    df["is_premium"] = df["customer_id"].map(seg["is_premium"]).fillna(False).astype(bool)

    # 6. Routing rules, checked in the order given in the brief
    def route(row):
        if row["status"] in HUMAN_REVIEW_STATUSES:
            return pd.Series({"route": "human_review", "priority": "NORMAL"})
        if row["category"] == "fraud":
            return pd.Series({"route": "security_team", "priority": "HIGH"})
        if row["is_premium"]:
            return pd.Series({"route": "premium_desk", "priority": "HIGH"})
        return pd.Series({"route": "support_queue", "priority": "NORMAL"})

    df[["route", "priority"]] = df.apply(route, axis=1)

    out = df[["ticket_id", "customer_id", "status", "category", "segment_name",
              "priority", "route", "tokens"]].rename(columns={"tokens": "tokens_used"})
    return out


def run_pipeline(system_prompt, tickets_df):
    mock_llm.reset()
    rows = []
    for _, row in tickets_df.iterrows():
        cleaned_text, _ = redact(row["text"])
        ok, reason = validate_input(cleaned_text)
        if not ok:
            rows.append({"ticket_id": row["ticket_id"], "true_category": row["true_category"],
                         "status": "rejected_input", "pred_category": None, "tokens": 0})
            continue
        resp = classify_with_retries(row["ticket_id"], cleaned_text, system_prompt)
        parsed = parse_reply(get_reply_text(resp))
        rows.append({"ticket_id": row["ticket_id"], "true_category": row["true_category"],
                     "status": parsed["status"], "pred_category": parsed["category"],
                     "tokens": get_total_tokens(resp)})
    return pd.DataFrame(rows)


def summarise(run_df, name):
    classified = run_df[run_df["status"].isin(["classified_json", "classified_keyword"])]
    n_classified = len(classified)
    correct = (classified["pred_category"] == classified["true_category"]).sum()
    total_tokens = run_df["tokens"].sum()
    print(f"--- {name} ---")
    print("Tickets classified:", n_classified)
    print(f"Accuracy: {correct}/{n_classified} = {correct / n_classified:.1%}")
    print("Total tokens:", total_tokens)
    print("Tokens per classified ticket:", round(total_tokens / n_classified, 1))
    print()
    return {"name": name, "classified": n_classified, "accuracy": correct / n_classified,
            "total_tokens": total_tokens, "tokens_per_ticket": total_tokens / n_classified}

def summarise_run(run_df, name):
    """Metrics for one full run. Accuracy is computed over classified tickets only."""
    classified = run_df[run_df["status"].isin(["classified_json", "classified_keyword"])]
    n = len(classified)
    correct = int((classified["pred_category"] == classified["true_category"]).sum())
    total_tokens = int(run_df["tokens"].sum())   # rejected tickets contribute 0
    return {
        "run": name,
        "tickets_classified": n,
        "correct": correct,
        "accuracy_pct": round(100 * correct / n, 1) if n else 0.0,
        "total_tokens": total_tokens,
        "tokens_per_classified": round(total_tokens / n, 1) if n else 0.0,
    }


def compare_prompts(ticket_df):
    """Run the full pipeline with each prompt and return the summary, confusion table and raw runs."""
    zero_df = run_pipeline(ZERO_SHOT_PROMPT, ticket_df)   # run_pipeline calls mock_llm.reset()
    few_df = run_pipeline(FEW_SHOT_PROMPT, ticket_df)

    summary = pd.DataFrame([
        summarise_run(zero_df, "zero-shot"),
        summarise_run(few_df, "few-shot"),
    ])

    # Confusion table for the zero-shot run, classified tickets only
    zero_classified = zero_df[zero_df["status"].isin(["classified_json", "classified_keyword"])]
    confusion = pd.crosstab(zero_classified["true_category"], zero_classified["pred_category"],
                            rownames=["true"], colnames=["predicted"])

    # List every zero-shot error by (true, predicted) pair
    errors = zero_classified[zero_classified["pred_category"] != zero_classified["true_category"]]
    error_pairs = errors.groupby(["true_category", "pred_category"]).size()

    return summary, confusion, error_pairs, zero_df, few_df

resp = mock_llm.chat([
    {"role": "system", "content": ZERO_SHOT_PROMPT},
    {"role": "user", "content": "I was charged twice"},
])
print(vars(resp))   # shows the attributes on the SimpleNamespace










if __name__ == "__main__":

    ticket_df = pd.read_csv('data/raw/tickets.csv')
    tickets = list(zip(ticket_df['ticket_id'], ticket_df['text']))
    print("\n\nnumber of raw tickets:",len(tickets))   # the number of tickets in the batch

    zero_df = run_pipeline(ZERO_SHOT_PROMPT, ticket_df)
    few_df = run_pipeline_budgeted(FEW_SHOT_PROMPT, ticket_df, TRIAGE_BUDGET)

    triage = build_triage_results(ticket_df)

    zero_summary = summarise(zero_df, "zero-shot")
    few_summary = summarise(few_df, "few-shot")

    summary, confusion, error_pairs, zero_df, few_df = compare_prompts(ticket_df)


    plan_transient_faults(tickets, target=19)

    transient_errors_injected = FAULT_PLAN["injected"]   # errors actually raised
    transient_errors_caught = FAULT_PLAN["caught"]       # errors caught by the retry logic

    print("\n\nTransientLLMErrors injected:", transient_errors_injected)
    print("TransientLLMErrors caught by retries:", transient_errors_caught)


    print(summary.to_string(index=False))
    print("\nZero-shot confusion table:")
    print(confusion)
    print("\nZero-shot error pairs:")
    print(error_pairs)

    # Confusion table for the zero-shot run (classified tickets only)
    zero_classified = zero_df[zero_df["status"].isin(["classified_json", "classified_keyword"])]
    confusion = pd.crosstab(zero_classified["true_category"], zero_classified["pred_category"], rownames=["true"], colnames=["predicted"])
    print(confusion)

    
    empty = sum(1 for _, t in tickets if not isinstance(t, str) or not str(t).strip())
    print("Empty tickets:", empty)
    
    results, rejected = run_all(tickets)

    statuses = []
    for tid, reply in results.items():
        text = get_reply_text(reply)
        statuses.append(parse_reply(text)["status"])

    results, rejected = run_all(tickets)

    print(Counter(statuses))
    print("Total tickets:", len(statuses)) 
    print("Rejected:", len(rejected))

    print(rejected)
    print(len(results), len(rejected))
    print(Counter(rejected.values()))

    counts = Counter(parse_reply(get_reply_text(resp))["status"] for resp in results.values())

    budget_used = TOKEN_BUDGET
    budget_df = run_pipeline_budgeted(ZERO_SHOT_PROMPT, ticket_df, budget_used)

    if (budget_df["status"] == "budget_exhausted").sum() == 0:
        budget_used = 5000
        budget_df = run_pipeline_budgeted(ZERO_SHOT_PROMPT, ticket_df, budget_used)

    budget_df.to_csv("outputs/triage_budget_demo.csv",index=False)

    triage.to_csv("outputs/triage_results.csv", index=False)

    exhausted = int((budget_df["status"] == "budget_exhausted").sum())
    print("\n\nBudget used:", budget_used)
    print("Tokens spent:", int(budget_df["tokens"].sum()))
    print("Status counts:\n", budget_df["status"].value_counts())
    print("Tickets budget_exhausted:", exhausted)


    print("Rows:", len(triage))   # should match the number of tickets (164)
    print(triage["route"].value_counts())
    print(triage["status"].value_counts())  