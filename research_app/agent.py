"""Implement the Part 1 research workflow as a LangGraph graph.

    plan ─▶ research_facet × 2–4 (parallel) ─▶ coverage ─▶ [followup] ─▶ synthesize ─▶ validate ─▶ [repair]

- plan: one LLM call splits the question into 2–4 subtopics.
- research_facet: per subtopic, one search, up to three fetches, one extract call.
- coverage: a deterministic check; the first subtopic with no evidence gets one
  follow-up branch.
- synthesize: one LLM call writes claims that cite evidence ids.
- validate: the supplied citation checker; at most one repair call.

Per report: at most 8 LLM calls and 20 tool operations (5 searches, 15
fetches), and at most 3 attempts per operation (the tracing client enforces
these). LLM calls share one global cap C across every running report.

Use the LLM stage names plan, extract, synthesize and repair, and build every
prompt with the matching `prompts.*_messages` function: FakeLLM recognizes
calls by stage and reads the prompt in that layout, and the fixed layout is
what makes calls share prefixes.

The workflow is adapted from LangChain's Deep Research from Scratch (MIT):
it plans the research, gathers evidence in parallel, and writes a report.
The supplied parser locates each quoted passage in the source text to obtain
its offsets; the model only needs to provide the quote.
"""

from __future__ import annotations

import asyncio
import operator
from dataclasses import dataclass, field
from typing import Annotated, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from . import prompts
from .artifacts import (BranchResult, BranchStatus, Claim, Evidence, Facet, Report, assign_evidence_ids, locate,
                        order_branches)
from .concurrency import AdjustableCap
from .corpus import Corpus, Doc, ToolLatency, Tools, terms
from .llm import LLM, _heuristic_facets
from .tracing import Recorder, Recording, TracingClient


@dataclass(frozen=True)
class AgentSettings:
    min_facets: int = 2
    max_facets: int = 4
    search_k: int = 5
    max_fetch: int = 3
    plan_tokens: int = 400
    extract_tokens: int = 700
    synthesize_tokens: int = 1200
    branch_timeout: float | None = None     # seconds per branch; None disables the timeout


@dataclass
class ReportContext:
    client: TracingClient
    question: dict                          # {"id", "question", ...} from questions.json
    settings: AgentSettings = field(default_factory=AgentSettings)

    @property
    def rid(self) -> str:
        return self.client.report_id

    @property
    def corpus(self) -> Corpus:
        return self.client.corpus


class ReportState(TypedDict, total=False):
    question: str
    question_id: str
    facets: list[Facet]
    plan_op: str
    branches: Annotated[list[BranchResult], operator.add]
    uncovered: int | None                   # index of the first subtopic without evidence
    evidence: list[Evidence]
    synth_messages: list[prompts.Msg]
    draft: str
    synth_op: str
    report: Report
    errors: list[str]
    validate_op: str
    repaired: bool
    valid: bool


def ctx_of(config: RunnableConfig) -> ReportContext:
    """Return the report context passed to this graph node through its configuration."""
    return config["configurable"]["ctx"]


# ---------------------------------------------------------------- parsing (supplied)


def _list_field(obj: dict, key: str) -> list:
    """Treat a missing or incorrectly typed JSON array as empty."""
    value = obj.get(key)
    return value if isinstance(value, list) else []


def parse_facets(text: str | None, question: str, settings: AgentSettings) -> list[Facet]:
    """Parse the planned subtopics, or split the question if the reply is unusable.

    The fallback produces a repeatable plan without another LLM call.
    """
    obj = prompts.parse_json(text or "") or {}
    raw = [f for f in _list_field(obj, "facets") if isinstance(f, dict) and f.get("title")]
    if len(raw) < settings.min_facets:
        raw = _heuristic_facets(question)
    raw = raw[:settings.max_facets]
    return [Facet(f"f{i + 1}", str(f["title"]).strip(), str(f.get("query") or f["title"]).strip())
            for i, f in enumerate(raw)]


def parse_evidence(text: str | None, docs: list[Doc], facet: Facet, call_id: str) -> list[Evidence]:
    """Parse quoted passages and locate them in the fetched documents.

    Discard quotes that cannot be matched to a source passage. Store offsets
    into the original document text so the citation checker can verify them.
    """
    obj = prompts.parse_json(text or "") or {}
    by_id = {d.doc_id: d for d in docs}
    out = []
    for item in _list_field(obj, "evidence"):
        if not isinstance(item, dict):
            continue
        doc = by_id.get(str(item.get("doc_id", "")))
        quote = str(item.get("quote", ""))
        span = locate(quote, doc.text) if doc else None
        if span is None:
            continue
        start, end = span
        out.append(Evidence("", doc.doc_id, doc.sha256, start, end, doc.text[start:end], facet.facet_id, doc.uri,
                            call_id))
    return out


def parse_report(text: str | None, ctx: ReportContext, evidence: list[Evidence]) -> Report:
    """Parse the model reply into a Report, omitting claims with no text."""
    obj = prompts.parse_json(text or "") or {}
    claims = [Claim(str(c.get("text", "")).strip(), [str(e) for e in _list_field(c, "evidence_ids")])
              for c in _list_field(obj, "claims") if isinstance(c, dict) and c.get("text")]
    return Report(ctx.rid, ctx.question["id"], ctx.question["question"], str(obj.get("title", "")), claims,
                  list(evidence))


# ---------------------------------------------------------------- graph nodes (Part 1 TODOs)


async def plan(state: ReportState, config: RunnableConfig) -> dict:
    """Plan the subtopics for one research question."""
    ctx = ctx_of(config)
    # [Part 1] TODO: Make the planning call and return the parsed subtopics.
    # Build the prompt from state["question"] and call the tracing client
    # with stage "plan", op_id f"{ctx.rid}/plan", no dependencies, and
    # order (0,). Pass the reply text to parse_facets, or None if the call
    # failed so the supplied fallback can produce a plan. Return the state
    # keys "facets" and "plan_op"; the latter is the operation's id.
    raise NotImplementedError("Part 1: fill plan")
    # END OF YOUR CODE


def fan_out(state: ReportState) -> list[Send]:
    """Create the parallel research tasks for the planned subtopics."""
    # [Part 1] TODO: Create one Send for each subtopic in plan order.
    # Each Send targets "research_facet" and carries a dict with "question",
    # "facet" (the Facet object), "index" (its zero-based position), and
    # "deps" (a list containing the plan operation's id). Return the list
    # of Send objects; LangGraph starts those branches in parallel.
    raise NotImplementedError("Part 1: fill fan_out")
    # END OF YOUR CODE


async def research_branch(ctx: ReportContext, question: str, facet: Facet, index: int, deps: list[str], *,
                          query: str | None = None, followup: bool = False) -> BranchResult:
    """Research one subtopic and retain the completed work if an operation fails."""
    # [Part 1] TODO: Search for sources, fetch them, and extract evidence.
    # Use query when provided, otherwise facet.query. Search once with
    # ctx.settings.search_k, then fetch up to ctx.settings.max_fetch hits
    # in parallel. Each fetch depends on the search. If any documents were
    # fetched, make one "extract" call using those documents. Record every
    # fetch as a dependency, including failed ones. Parse its reply
    # with parse_evidence; unusable quotes are discarded by that helper.
    #
    # Return a BranchResult with the document ids, operation ids, evidence,
    # status, and errors. Add each operation id before awaiting it and save
    # each successful document as it arrives, so a later timeout cannot
    # lose completed work. SUCCESS means no operation failed; PARTIAL means
    # some fetches failed but extraction succeeded. Use FAILED for a failed
    # search or extraction, no search hits, or no fetched documents. Empty
    # evidence alone does not mean the extraction operation failed.
    #
    # Use asyncio.timeout when branch_timeout is set. Catch TimeoutError,
    # mark the branch TIMEOUT, and return the completed work. Regular
    # branches use facet.facet_id in their operation ids, index in their
    # BranchResult, and orders (1, index, ...). A follow-up uses "followup",
    # 100 + index, and orders (2, 0, ...). Within either branch, search is
    # step (0,), fetch k is step (1, k), and extraction is step (2,).
    raise NotImplementedError("Part 1: fill research_branch")
    # END OF YOUR CODE


async def research_facet(state: dict, config: RunnableConfig) -> dict:
    """Run one parallel branch and return its result to the graph."""
    ctx = ctx_of(config)
    # [Part 1] TODO: Run the research task described by this Send payload.
    # Pass its question, Facet, index, and dependencies to research_branch.
    # Return {"branches": [result]} so the branches reducer can concatenate
    # this result with the other branches. Do not replace the shared list
    # with a single BranchResult or mutate another branch's state.
    raise NotImplementedError("Part 1: fill research_facet")
    # END OF YOUR CODE


def coverage(state: ReportState) -> dict:
    """Find the first planned subtopic for which no evidence was collected."""
    # [Part 1] TODO: Check which subtopics have evidence after the branches join.
    # A subtopic is covered if its original research branch returned at
    # least one usable evidence item. Examine subtopics in plan order and
    # return {"uncovered": index} for the first missing one, or None as the
    # value if all are covered. The value is an index, not a Facet object.
    raise NotImplementedError("Part 1: fill coverage")
    # END OF YOUR CODE


def route_coverage(state: ReportState) -> str:
    """Choose whether to research an uncovered subtopic or write the report."""
    # [Part 1] TODO: Choose the next node from the coverage result.
    # Return "followup" if "uncovered" contains an index; otherwise return
    # "synthesize". Index 0 identifies the first subtopic and must trigger
    # a follow-up too, so a truth-value check on the index is insufficient.
    raise NotImplementedError("Part 1: fill route_coverage")
    # END OF YOUR CODE


def _branch_tails(branches: list[BranchResult]) -> list[str]:
    """Collect the operation ids that the next research step must wait for.

    A failed or timed-out branch may have started several parallel operations.
    Include all of them to preserve those dependencies. This also includes
    some earlier operations whose completion is already implied by later ones.
    """
    return list(dict.fromkeys(op for b in order_branches(branches) for op in b.ops))


async def followup(state: ReportState, config: RunnableConfig) -> dict:
    """Make one additional research attempt for the first uncovered subtopic."""
    ctx = ctx_of(config)
    # [Part 1] TODO: Research the missing subtopic once more with a broader query.
    # Look up the subtopic at state["uncovered"]. Build a query from its
    # title, original query, and the report question. Run research_branch
    # with followup=True and dependencies from _branch_tails so it waits
    # for all original branches. Return the result as a one-element list
    # under "branches", just as research_facet does.
    raise NotImplementedError("Part 1: fill followup")
    # END OF YOUR CODE


async def synthesize(state: ReportState, config: RunnableConfig) -> dict:
    """Combine the collected evidence into a report with citations."""
    ctx = ctx_of(config)
    # [Part 1] TODO: Number the evidence and make the synthesis call.
    # Use assign_evidence_ids to combine the branches in a stable order,
    # then build the prompt from the question, plan, and numbered evidence.
    # The "synthesize" call uses op_id f"{ctx.rid}/synthesize", order (3,),
    # and dependencies from _branch_tails(state["branches"]).
    # Return "evidence", "synth_messages", "draft", "synth_op", and "report".
    # Keep the messages and raw reply for a possible repair; "report" is
    # parse_report's result. If the call failed, use an empty draft and
    # pass None to parse_report so validation can report the missing claims.
    raise NotImplementedError("Part 1: fill synthesize")
    # END OF YOUR CODE


async def validate(state: ReportState, config: RunnableConfig) -> dict:
    """Check the report citations and save the validation result."""
    ctx = ctx_of(config)
    # [Part 1] TODO: Run the supplied citation checker through the tracing client.
    # This is a synchronous call, so do not await it. Before repair, use
    # op_id f"{ctx.rid}/validate" and order (4,); afterward use
    # f"{ctx.rid}/revalidate" and order (6,). Both depend on state["synth_op"].
    # Leave snapshot unset so evidence can cite either saved corpus version.
    # Return "errors" as a list of strings, "validate_op" as the operation
    # id, and "valid" as True only when no errors were found.
    raise NotImplementedError("Part 1: fill validate")
    # END OF YOUR CODE


def route_validation(state: ReportState) -> str:
    """End the report or allow its one repair attempt."""
    # [Part 1] TODO: Decide whether another LLM call is needed.
    # Return END if the report is valid, a repair has already been attempted,
    # or there is no draft to repair. Otherwise return "repair". A report
    # that still fails after repair must finish with valid=False; do not
    # route it through another repair attempt.
    raise NotImplementedError("Part 1: fill route_validation")
    # END OF YOUR CODE


async def repair(state: ReportState, config: RunnableConfig) -> dict:
    """Ask the model to fix citation errors in the current draft."""
    ctx = ctx_of(config)
    # [Part 1] TODO: Continue the synthesis conversation with the checker's errors.
    # Build repair_messages from the saved synthesis messages, draft, and
    # errors. Make one "repair" call with op_id f"{ctx.rid}/repair",
    # order (5,), and the validation operation as its dependency. Parse
    # a successful reply using the existing evidence; if the call fails,
    # keep the current report. Return "report", "repaired": True, and
    # "synth_op" set to the repair operation id so revalidation waits for it.
    # Mark the attempt as repaired even if the call failed.
    raise NotImplementedError("Part 1: fill repair")
    # END OF YOUR CODE


def build_graph():
    """Build and compile the required LangGraph workflow."""
    # [Part 1] TODO: Connect the graph nodes and return the compiled graph.
    # Create a StateGraph using ReportState and register the seven nodes
    # above. START leads to plan; fan_out sends the planned subtopics to
    # research_facet. Join those branches at coverage, then use
    # route_coverage to choose followup or synthesize. Followup also leads
    # to synthesize. Synthesis leads to validate, where route_validation
    # chooses repair or END. Repair returns to validate for the final check.
    raise NotImplementedError("Part 1: fill build_graph")
    # END OF YOUR CODE


# ---------------------------------------------------------------- running (supplied)

_GRAPH = None


async def run_report(ctx: ReportContext) -> ReportState:
    """Run the compiled graph for one report and return its final state."""
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_graph()
    state = await _GRAPH.ainvoke({"question": ctx.question["question"], "question_id": ctx.question["id"]},
                                 config={"configurable": {"ctx": ctx}, "recursion_limit": 50})
    rec = ctx.client.rec
    rec.valid = state.get("valid", False)
    rec.errors = state.get("errors", [])
    rec.report = state["report"].to_dict() if state.get("report") else None
    rec.branches = [{"facet_id": b.facet.facet_id, "index": b.index, "followup": b.followup, "status": b.status.value,
                     "docs": b.docs, "evidence": len(b.evidence), "errors": b.errors}
                    for b in order_branches(state.get("branches", []))]
    return state


async def run_reports(questions: list[dict], *, llm: LLM, cap: AdjustableCap, arrivals: list[float] | None = None,
                      snapshot: str = "v2", live: bool = True, latency: ToolLatency | None = None, faults=None,
                      settings: AgentSettings = AgentSettings(), backend: str = "",
                      recorder: Recorder | None = None) -> tuple[Recorder, list[Recording]]:
    """Run one report per question, sharing the same concurrency cap.

    In live mode, report i starts arrivals[i] seconds after the run begins.
    Return the recorder and a list of recordings in question order.
    """
    recorder = recorder or Recorder(cap=cap, live=live, backend=backend)
    tools = Tools(Corpus(snapshot), latency, faults)
    arrivals = arrivals or [0.0] * len(questions)

    async def one(i: int, q: dict, arrival: float) -> Recording:
        if live:
            await asyncio.sleep(max(0.0, arrival - recorder.now()))
        client = recorder.open_report(f"R{i:02d}", q, arrival=arrival, snapshot=snapshot, llm=llm, tools=tools)
        try:
            await run_report(ReportContext(client, q, settings))
        finally:
            recorder.close_report(client.rec)
        return client.rec

    recs = await asyncio.gather(*(one(i, q, a) for i, (q, a) in enumerate(zip(questions, arrivals))))
    return recorder, list(recs)
