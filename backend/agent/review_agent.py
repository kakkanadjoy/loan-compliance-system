"""Review agent — a LangGraph workflow that takes one loan through review.

The graph (see also the diagram in the project docs):

    START -> gather -> [conditional branch on routing]
                         fast_track          -> auto_approve
                         manager/senior band -> waiver_analysis
                         compliance override -> compliance_escalation
                       all paths -> draft_memo -> END

Design principles carried over from the rest of the system:
- The agent NEVER re-decides compliance. The rules engine (via the explain
  pipeline) is the sole source of findings and routing; the agent
  orchestrates procedure around those facts.
- Deterministic by default. The memo drafter is a template function; an LLM
  (e.g. Bedrock via langchain-aws) can be plugged in via `memo_drafter` to
  polish prose, but facts always come from the rules engine.
- Observable. Every node appends to state["path"], so any run can show
  exactly which route a loan took — audit-friendly, and easy to test.

CLI:
    python agent/review_agent.py APP-2024-0712
    python agent/review_agent.py APP-2024-0291 --json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Annotated, Callable, TypedDict

from langgraph.graph import END, START, StateGraph

sys.path.append(str(Path(__file__).resolve().parents[1]))
from app import rules_engine  # noqa: E402
from rag.explain import FileCorpusStore, explain_record, load_record, open_store  # noqa: E402


# --------------------------------------------------------------------------
# State — the single object that flows through the graph
# --------------------------------------------------------------------------

def _append(existing: list, new: list) -> list:
    return existing + new


class ReviewState(TypedDict, total=False):
    record: dict                       # input: the loan
    explanation: dict                  # gather: explain pipeline output
    decision: str                      # auto_approved | review_required | escalated_to_compliance
    queue: str                         # which human queue, if any
    minimum_authority: str | None      # lowest level that can waive everything
    blocking_codes: list[str]          # unwaivable exception codes
    memo: str                          # the written decision
    path: Annotated[list[str], _append]  # trace of nodes visited


# --------------------------------------------------------------------------
# Nodes
# --------------------------------------------------------------------------

def make_gather(store):
    def gather(state: ReviewState) -> dict:
        explanation = explain_record(state["record"], store=store)
        return {"explanation": explanation, "path": ["gather"]}

    return gather


def route_after_gather(state: ReviewState) -> str:
    """The conditional edge: which procedure does this loan get?"""
    routing = state["explanation"]["routing"]
    if routing == "fast_track":
        return "auto_approve"
    if routing == "compliance_review":
        return "compliance_escalation"
    return "waiver_analysis"  # manager_review or senior_review


def auto_approve(state: ReviewState) -> dict:
    return {
        "decision": "auto_approved",
        "queue": "none",
        "minimum_authority": None,
        "blocking_codes": [],
        "path": ["auto_approve"],
    }


def waiver_analysis(state: ReviewState) -> dict:
    """Who is the lowest authority level that could waive every finding?"""
    findings = state["explanation"]["findings"]
    needed_idx = -1
    for f in findings:
        # waivable_by is ascending; its first entry is the minimum level
        # for that finding. The loan needs the max of those minimums.
        idx = rules_engine.AUTHORITY_ORDER.index(f["waivable_by"][0])
        needed_idx = max(needed_idx, idx)
    return {
        "decision": "review_required",
        "queue": state["explanation"]["routing"],
        "minimum_authority": rules_engine.AUTHORITY_ORDER[needed_idx],
        "blocking_codes": [],
        "path": ["waiver_analysis"],
    }


def compliance_escalation(state: ReviewState) -> dict:
    findings = state["explanation"]["findings"]
    blocking = [f["exception_code"] for f in findings if not f["waivable_by"]]
    return {
        "decision": "escalated_to_compliance",
        "queue": "compliance_review",
        "minimum_authority": None,   # no level suffices — that's the point
        "blocking_codes": blocking,
        "path": ["compliance_escalation"],
    }


# --- memo drafting: deterministic template, LLM-swappable ------------------

def template_memo(state: ReviewState) -> str:
    e = state["explanation"]
    lines = [
        f"REVIEW MEMO — {e['loan_id']} ({e['loan_type']}, ${e['loan_amount']:,.0f})",
        f"Risk score {e['risk_score']} | max severity {e['max_severity']} | "
        f"decision: {state['decision'].upper()}",
        "",
    ]
    if state["decision"] == "auto_approved":
        lines.append(
            "No policy exceptions identified. Eligible for fast-track per "
            "routing thresholds. No human sign-off required."
        )
    elif state["decision"] == "review_required":
        lines.append(
            f"{e['exception_count']} exception(s); queue: {state['queue']}. "
            f"Minimum waiver authority required: {state['minimum_authority']}."
        )
        for f in e["findings"]:
            lines.append(
                f"  - {f['exception_code']} (sec {f['policy_section']}, "
                f"{f['severity']}): observed {f['observed']} vs "
                f"{f['threshold']}; waivable by {f['waivable_by'][0]}+"
            )
    else:  # escalated_to_compliance
        lines.append(
            f"UNWAIVABLE finding(s) {', '.join(state['blocking_codes'])} — "
            "routed to compliance officer. No credit authority may clear "
            "this file."
        )
        for f in e["findings"]:
            tag = "BLOCKING" if not f["waivable_by"] else f["severity"]
            lines.append(
                f"  - {f['exception_code']} (sec {f['policy_section']}, {tag}): "
                f"observed {f['observed']} vs {f['threshold']}"
            )
    lines.append("")
    lines.append("Facts per rules engine; citations per policy corpus. "
                 "Generated by review agent.")
    return "\n".join(lines)


def make_draft_memo(memo_drafter: Callable[[ReviewState], str]):
    def draft_memo(state: ReviewState) -> dict:
        return {"memo": memo_drafter(state), "path": ["draft_memo"]}

    return draft_memo


# --------------------------------------------------------------------------
# Graph assembly
# --------------------------------------------------------------------------

def build_graph(store=None, memo_drafter: Callable[[ReviewState], str] = template_memo):
    """Compile the review workflow. `store` is a policy store (db/file);
    `memo_drafter` is where an LLM could be plugged in later."""
    store = store or FileCorpusStore()
    g = StateGraph(ReviewState)
    g.add_node("gather", make_gather(store))
    g.add_node("auto_approve", auto_approve)
    g.add_node("waiver_analysis", waiver_analysis)
    g.add_node("compliance_escalation", compliance_escalation)
    g.add_node("draft_memo", make_draft_memo(memo_drafter))

    g.add_edge(START, "gather")
    g.add_conditional_edges("gather", route_after_gather)
    g.add_edge("auto_approve", "draft_memo")
    g.add_edge("waiver_analysis", "draft_memo")
    g.add_edge("compliance_escalation", "draft_memo")
    g.add_edge("draft_memo", END)
    return g.compile()


def review_loan(record: dict, store=None) -> ReviewState:
    graph = build_graph(store=store)
    return graph.invoke({"record": record, "path": []})


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("loan_id")
    parser.add_argument("--store", choices=["auto", "db", "file"], default="auto")
    parser.add_argument("--json", action="store_true", dest="as_json")
    args = parser.parse_args()

    record = load_record(args.loan_id)
    store, used = open_store(args.store)

    from agent.llm_memo import llm_available, llm_memo_drafter
    if llm_available():
        drafter, drafter_name = llm_memo_drafter, "azure llm"
    else:
        drafter, drafter_name = template_memo, "template"
    graph = build_graph(store=store, memo_drafter=drafter)
    final = graph.invoke({"record": record, "path": []})

    if args.as_json:
        out = {k: v for k, v in final.items() if k != "record"}
        print(json.dumps(out, indent=2))
        return
    print("path: " + " -> ".join(final["path"]))
    print()
    print(final["memo"])
    print(f"\n(policy store: {used} | memo drafter: {drafter_name})")


if __name__ == "__main__":
    main()
