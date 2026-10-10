import re
import json
import html
import time
import operator
import tempfile
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal, TypedDict

import gradio as gr
from langgraph.graph import StateGraph, START, END
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

LOCAL = "Local (Ollama)"
ONLINE = "Online (Cloud APIs)"
MAX_ROUNDS = 2
REQUEST_TIMEOUT = 180   # seconds; one stuck call can no longer freeze the whole app

# Online mode works with any OpenAI-compatible provider.
# OpenAI (paid):  ONLINE_BASE_URL = None,  ONLINE_MODEL = "gpt-4o-mini"
# Groq (free tier): ONLINE_BASE_URL = "https://api.groq.com/openai/v1",  ONLINE_MODEL = "llama-3.1-8b-instant"
ONLINE_BASE_URL = None
ONLINE_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# 1. Agent configuration (the single place to edit names, prompts, models)
# ---------------------------------------------------------------------------
AGENTS = {
    "id": {
        "title": "ID (INSTINCT / PRAGMATIST)",
        "local_model": "qwen2.5:1.5b",
        "temperature": 0.5,
        "max_tokens": 512,
        "system": """You are ID, the instinctive drive of the T.R.I.A.D. system.
Your focus is real-world practical execution, speed, cost-efficiency, human desires and psychology, and compromise.
Focus on what actually works in practice, even if messy. Keep answers under 250 words.""",
        "task": "Provide your practical execution strategy.",
        "critique_style": "pragmatically",
    },
    "ego": {
        "title": "EGO (RATIONALIST)",
        "local_model": "deepseek-r1:1.5b",
        "temperature": 0.2,
        "max_tokens": 2048,   # higher: thinking tokens count towards this limit
        "thinks": True,       # reasoning model: its thinking is captured separately
        "system": """You are EGO, the rational core of the T.R.I.A.D. system.
Your focus is empirical evidence, logical deduction, computational rigor, and mathematical feasibility.
Ignore emotional or ethical considerations unless specifically asked. Be direct and uncompromising on hard facts.
Keep answers under 250 words.""",
        "task": "Provide your logical, evidence-based solution.",
        "critique_style": "logically",
    },
    "superego": {
        "title": "SUPEREGO (CONSCIENCE)",
        "local_model": "llama3.2:1b",
        "temperature": 0.3,
        "max_tokens": 512,
        "system": """You are SUPEREGO, the moral conscience of the T.R.I.A.D. system.
Your focus is human safety, risk mitigation, ethics, long-term stability, and empathy.
Analyze all proposals for unintended consequences, ethical pitfalls, and risks to wellbeing.
Keep answers under 250 words.""",
        "task": "Provide your safety and ethically focused solution.",
        "critique_style": "ethically",
    },
}

# The neutral judge writes the final answer. It is not one of the three
# debaters, so no single perspective dominates the consensus.
JUDGE = {
    "local_model": "llama3.2:3b",
    "temperature": 0.2,
    "max_tokens": 1024,
    "system": "You are the T.R.I.A.D. Consensus Engine, a neutral judge. "
              "Synthesize the 3 positions into a final objective answer. Keep it under 300 words.",
}

# Shown in the status bar as each graph step finishes
STEP_NAMES = {
    "id_proposal": "Id proposal ready",
    "ego_proposal": "Ego proposal ready",
    "superego_proposal": "Superego proposal ready",
    "cross_critique": "Cross-critique finished",
    "voting": "Votes counted",
    "revision": "Proposals revised",
    "consensus": "Consensus synthesized",
}


# ---------------------------------------------------------------------------
# 2. State
# ---------------------------------------------------------------------------
class TriadState(TypedDict):
    query: str
    mode: str
    api_key: str
    ego_thought: str
    # operator.or_ merges the dicts written by the 3 parallel proposal nodes
    proposals: Annotated[dict, operator.or_]
    critiques: dict
    votes: dict
    # operator.add appends one entry per debate round: {"critiques": ..., "votes": ..., "reasons": ...}
    history: Annotated[list, operator.add]
    iteration: int
    final_consensus: str


# ---------------------------------------------------------------------------
# 3. Helpers
# ---------------------------------------------------------------------------
def split_thinking(text: str) -> tuple[str, str]:
    """For models that put <think>...</think> inside the answer text (e.g. via cloud APIs)."""
    if "</think>" in text:
        thought, answer = text.split("</think>", 1)
    elif "<think>" in text:  # ran out of tokens while still thinking
        thought, answer = text, ""
    else:
        thought, answer = "", text
    return thought.replace("<think>", "").strip(), answer.strip()


def truncate(text: str, max_chars: int = 1200) -> str:
    return text if len(text) <= max_chars else text[:max_chars] + "\n...[truncated]..."


def format_block(title: str, texts: dict) -> str:
    lines = [f"--- {title} ---"]
    lines += [f"{agent.upper()}: {truncate(text)}" for agent, text in texts.items()]
    return "\n".join(lines)


def agent_config(agent: str) -> dict:
    """Settings for one of the 3 debaters, or for the neutral judge."""
    return JUDGE if agent == "judge" else AGENTS[agent]


@lru_cache(maxsize=None)
def get_model(agent: str, mode: str, api_key: str, think: bool = True):
    """Create each model once and reuse it (cached per agent/mode/key/think)."""
    cfg = agent_config(agent)
    if mode == LOCAL:
        return ChatOllama(
            model=cfg["local_model"],
            temperature=cfg["temperature"],
            # True: thinking returned separately; False: no thinking; None: model not a thinker
            reasoning=think if cfg.get("thinks") else None,
            num_ctx=4096,
            num_predict=cfg["max_tokens"],          # hard cap on output length
            client_kwargs={"timeout": REQUEST_TIMEOUT},
        )
    return ChatOpenAI(
        model=ONLINE_MODEL,
        base_url=ONLINE_BASE_URL,
        temperature=cfg["temperature"],
        api_key=api_key,
        max_tokens=cfg["max_tokens"],
        timeout=REQUEST_TIMEOUT,
        max_retries=1,
    )


def ask(state: TriadState, agent: str, prompt: str, system: str = "") -> tuple[str, str]:
    """Send one prompt to one agent. Returns (thought, answer)."""
    cfg = agent_config(agent)
    messages = [
        SystemMessage(content=system or cfg["system"]),
        HumanMessage(content=prompt),
    ]
    reply = get_model(agent, state["mode"], state["api_key"]).invoke(messages)
    tag_thought, answer = split_thinking(reply.content)
    thought = reply.additional_kwargs.get("reasoning_content", "") or tag_thought

    # Thinking used up the whole token budget: ask again with thinking switched off
    if not answer and cfg.get("thinks") and state["mode"] == LOCAL:
        reply = get_model(agent, state["mode"], state["api_key"], think=False).invoke(messages)
        answer = split_thinking(reply.content)[1]
        thought += "\n\n[Thinking hit the token limit, so the answer was generated with thinking off.]"

    return thought or "Direct logical processing.", answer or "(No answer returned by the model.)"


def ask_all(state: TriadState, prompts: dict) -> dict:
    """Ask every agent its own prompt at the same time. Returns {agent: answer}."""
    with ThreadPoolExecutor(max_workers=len(prompts)) as pool:
        futures = {agent: pool.submit(ask, state, agent, p) for agent, p in prompts.items()}
        return {agent: f.result()[1] for agent, f in futures.items()}


def parse_vote(text: str) -> str:
    """Look for 'VOTE: X' first. If the model ignored the format,
    take the first whole-word vote found. Default to MODIFY."""
    match = re.search(r"VOTE:\s*(APPROVE|MODIFY|REJECT)", text.upper())
    if match:
        return match.group(1)
    found = re.findall(r"\b(APPROVE|MODIFY|REJECT)\b", text.upper())
    return found[0] if found else "MODIFY"


def parse_reason(text: str) -> str:
    """Return the text after 'REASON:' (one line), or an empty string."""
    match = re.search(r"REASON:\s*(.+)", text, flags=re.IGNORECASE)
    return match.group(1).strip() if match else ""


def save_debate(state: dict) -> str:
    """Write the debate to a JSON file and return its path. The API key is left out."""
    data = {key: value for key, value in state.items() if key != "api_key"}
    path = Path(tempfile.gettempdir()) / f"triad_debate_{time.strftime('%Y%m%d_%H%M%S')}.json"
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# 4. Graph nodes
# ---------------------------------------------------------------------------
def make_proposal_node(agent: str):
    def node(state: TriadState) -> dict:
        thought, answer = ask(state, agent, f"User Query: {state['query']}\n{AGENTS[agent]['task']}")
        update = {"proposals": {agent: answer}}
        if agent == "ego":  # only the reasoning model produces a <think> trace
            update["ego_thought"] = thought
        return update
    return node


def cross_critique_node(state: TriadState) -> dict:
    if state.get("critiques"):  # 2nd round: proposals were revised, show the old critiques too
        context = format_block("REVISED PROPOSALS", state["proposals"])
        context += "\n\n" + format_block("PREVIOUS ROUND CRITIQUES", state["critiques"])
    else:
        context = format_block("PROPOSALS", state["proposals"])

    prompts = {}
    for agent, cfg in AGENTS.items():
        others = " and ".join(a.upper() for a in AGENTS if a != agent)
        prompts[agent] = f"{context}\n\nCritique {others} {cfg['critique_style']}."
    return {"critiques": ask_all(state, prompts)}


def voting_node(state: TriadState) -> dict:
    prompt = (
        f"Query: {state['query']}\n\n"
        f"{format_block('CRITIQUES & DEBATE', state['critiques'])}\n\n"
        "Cast your vote. Reply in exactly this format, on two lines:\n"
        "VOTE: APPROVE, MODIFY or REJECT\n"
        "REASON: one short sentence."
    )
    answers = ask_all(state, {agent: prompt for agent in AGENTS})
    votes = {agent: parse_vote(answer) for agent, answer in answers.items()}
    reasons = {agent: parse_reason(answer) for agent, answer in answers.items()}
    return {
        "votes": votes,
        "iteration": state.get("iteration", 0) + 1,
        "history": [{"critiques": state["critiques"], "votes": votes, "reasons": reasons}],
    }


def revision_node(state: TriadState) -> dict:
    """No majority: each agent rewrites its own proposal using the critiques."""
    prompts = {}
    for agent in AGENTS:
        prompts[agent] = (
            f"Query: {state['query']}\n\n"
            f"Your proposal:\n{truncate(state['proposals'][agent])}\n\n"
            f"{format_block('CRITIQUES', state['critiques'])}\n\n"
            "Rewrite your proposal, fixing the fair points raised against it. "
            "Keep your own perspective. Keep it under 250 words."
        )
    return {"proposals": ask_all(state, prompts)}


def consensus_node(state: TriadState) -> dict:
    prompt = "\n\n".join([
        f"Query: {state['query']}",
        format_block("PROPOSALS", state["proposals"]),
        format_block("CRITIQUES", state["critiques"]),
        f"Votes: {json.dumps(state['votes'])}",
    ])
    return {"final_consensus": ask(state, "judge", prompt)[1]}


def route_after_vote(state: TriadState) -> Literal["consensus", "revision"]:
    approvals = list(state["votes"].values()).count("APPROVE")
    if approvals >= 2 or state["iteration"] >= MAX_ROUNDS:
        return "consensus"
    return "revision"


# ---------------------------------------------------------------------------
# 5. Graph assembly
# ---------------------------------------------------------------------------
def build_graph():
    builder = StateGraph(TriadState)
    builder.add_node("cross_critique", cross_critique_node)
    builder.add_node("voting", voting_node)
    builder.add_node("revision", revision_node)
    builder.add_node("consensus", consensus_node)

    for agent in AGENTS:  # 3 proposals run in parallel
        name = f"{agent}_proposal"
        builder.add_node(name, make_proposal_node(agent))
        builder.add_edge(START, name)
        builder.add_edge(name, "cross_critique")

    builder.add_edge("cross_critique", "voting")
    builder.add_conditional_edges("voting", route_after_vote)
    builder.add_edge("revision", "cross_critique")   # revised proposals get criticized again
    builder.add_edge("consensus", END)
    return builder.compile()


app_graph = build_graph()


# ---------------------------------------------------------------------------
# 6. Debate log rendering
# ---------------------------------------------------------------------------
def round_outcome(votes: dict, round_no: int) -> str:
    approvals = list(votes.values()).count("APPROVE")
    if approvals >= 2:
        return "Majority approved. Consensus synthesized."
    if round_no >= MAX_ROUNDS:
        return "No majority. Round limit reached, consensus synthesized anyway."
    return "No majority. Agents revise their proposals and the debate continues."


def render_debate_log(history: list) -> str:
    """Turn the round history into HTML: one block per round, one card per agent."""
    if not history:
        return "<p class='crit-empty'>Run a query to see how the agents challenged each other.</p>"

    rounds = []
    for round_no, rnd in enumerate(history, start=1):
        votes = rnd["votes"]
        tally = ", ".join(f"{v.title()} {list(votes.values()).count(v)}" for v in ("APPROVE", "MODIFY", "REJECT"))

        cards = []
        for agent in AGENTS:
            vote = votes.get(agent, "MODIFY")
            targets = " and ".join(a.title() for a in AGENTS if a != agent)
            text = html.escape(rnd["critiques"].get(agent, ""))
            reason = html.escape(rnd.get("reasons", {}).get(agent, ""))
            reason_html = f'<div class="crit-reason">Why {vote}: {reason}</div>' if reason else ""
            cards.append(f"""
            <div class="crit-card crit-{agent}">
              <div class="crit-head">
                <span class="crit-who"><b>{agent.title()}</b> on {targets}</span>
                <span class="crit-vote vote-{vote.lower()}">{vote}</span>
              </div>
              <div class="crit-body">{text}</div>
              {reason_html}
            </div>""")

        cards_html = "".join(cards)
        rounds.append(f"""
        <section class="crit-round">
          <div class="crit-round-head">
            <span class="crit-round-no">Round {round_no}</span>
            <span class="crit-tally">{tally}</span>
          </div>
          <div class="crit-grid">{cards_html}</div>
          <p class="crit-outcome">{round_outcome(votes, round_no)}</p>
        </section>""")

    return "".join(rounds)


def render_outputs(state: dict, status: str, debate_file: str | None = None) -> tuple:
    """Build every dashboard output from whatever the graph has produced so far."""
    votes = state.get("votes") or {}
    proposals = state.get("proposals") or {}
    votes_display = "  |  ".join(f"{a.upper()}: [{votes.get(a, 'PENDING')}]" for a in AGENTS)
    return (
        f"*{status}*",
        state.get("final_consensus", ""),
        votes_display,
        state.get("ego_thought", ""),
        proposals.get("id", ""),
        proposals.get("ego", ""),
        proposals.get("superego", ""),
        render_debate_log(state.get("history", [])),
        debate_file,
    )


# ---------------------------------------------------------------------------
# 7. Gradio interface
# ---------------------------------------------------------------------------
def toggle_api_key_visibility(mode):
    return gr.update(visible=(mode == ONLINE))


def launch_triad_system(mode, api_key):
    # Gradio 6 sends None for empty or hidden fields, so fall back to safe defaults
    mode = mode or LOCAL
    api_key = (api_key or "").strip()
    if mode == ONLINE and not api_key:
        raise gr.Error("API Key is required for Online Cloud Mode!")
    status = f"SYSTEM ONLINE // MODE: {mode.upper()}"
    return gr.update(visible=False), gr.update(visible=True), mode, api_key, status


def execute_triad_query(query, mode, api_key):
    """Generator: pushes partial results to the page after every graph step."""
    query, mode, api_key = (query or "").strip(), mode or LOCAL, api_key or ""
    if not query:
        raise gr.Error("Please enter a valid query.")

    start = time.time()
    state, status = {}, "Debate started. Waiting for the first proposals..."
    yield render_outputs(state, status)

    inputs = {"query": query, "mode": mode, "api_key": api_key, "iteration": 0}
    try:
        for kind, chunk in app_graph.stream(inputs, stream_mode=["updates", "values"]):
            if kind == "values":
                state = chunk
            else:
                done = ", ".join(STEP_NAMES.get(node, node) for node in chunk)
                status = f"[{time.time() - start:.0f}s] {done}"
            yield render_outputs(state, status)
    except Exception as e:
        hint = " Is Ollama running, and are all four models pulled (see README)?" if mode == LOCAL else ""
        raise gr.Error(f"Debate stopped: {type(e).__name__}: {e}.{hint}")

    status = f"Done in {time.time() - start:.0f}s // MODE: {mode.upper()}"
    yield render_outputs(state, status, save_debate(state))


# Read the CSS ourselves so it works no matter which folder the app is started from
CSS = (Path(__file__).parent / "style.css").read_text(encoding="utf-8")

with gr.Blocks(title="T.R.I.A.D. DECISION ENGINE") as demo:
    active_mode = gr.State(LOCAL)
    active_key = gr.State("")

    # --- SCREEN 1: SPLASH SCREEN ---
    with gr.Column(visible=True, elem_classes=["splash-container"]) as splash_screen:
        gr.Markdown("# ── T.R.I.A.D. DECISION ENGINE INITIALIZATION ──")
        gr.Markdown("### SELECT SUPERCOMPUTER EXECUTION ENGINE")
        mode_radio = gr.Radio(choices=[LOCAL, ONLINE], value=LOCAL, label="EXECUTION ENGINE MODE")
        api_key_input = gr.Textbox(
            label="ENTER CLOUD API KEY (OpenAI / Compatible)",
            placeholder="sk-...", type="password", visible=False,
        )
        init_btn = gr.Button("INITIALIZE T.R.I.A.D. DECISION ENGINE", variant="primary", size="lg")

    # --- SCREEN 2: MAIN DASHBOARD ---
    with gr.Column(visible=False) as main_dashboard:
        gr.Markdown("# ── T.R.I.A.D. DECISION ENGINE ──")
        gr.Markdown("*SYNTHETIC NETWORK FOR OBJECTIVE DELIBERATION*")
        gr.Markdown("*T.R.I.A.D. — Tripartite Rational Inquiry & Automated Decisioning*")
        system_status_bar = gr.Markdown("*SYSTEM INITIALIZING...*")

        with gr.Row():
            user_input = gr.Textbox(
                label="ENTER SYSTEM QUERY / PROBLEM STATEMENT",
                placeholder="e.g., Should healthcare decisions be completely automated by AI?",
                lines=2, scale=4,
            )
            submit_btn = gr.Button("EXECUTE T.R.I.A.D. DEBATE", variant="primary", scale=1)

        vote_output = gr.Textbox(label="DECISION STATUS & VOTING MATRIX", interactive=False, elem_classes=["vote-box"])
        final_output = gr.Textbox(label="FINAL SYNTHESIZED CONSENSUS", lines=8, interactive=False)
        debate_file = gr.File(label="DOWNLOAD THIS DEBATE (JSON)", interactive=False)

        with gr.Accordion("DEBATE LOG: HOW EACH ROUND WAS ARGUED AND VOTED", open=True):
            crit_output = gr.HTML(render_debate_log([]))

        with gr.Accordion("VIEW AGENT THOUGHT PROCESSES & INDIVIDUAL PROPOSALS", open=True):
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### {AGENTS['id']['title']}")
                    id_prop = gr.Textbox(label="Pragmatic Proposal", lines=10, interactive=False)
                with gr.Column():
                    gr.Markdown(f"### {AGENTS['ego']['title']}")
                    ego_thought = gr.Textbox(label="Reasoning Trace (<think>)", lines=4, interactive=False)
                    ego_prop = gr.Textbox(label="Logical Proposal", lines=6, interactive=False)
                with gr.Column():
                    gr.Markdown(f"### {AGENTS['superego']['title']}")
                    superego_prop = gr.Textbox(label="Ethical / Safety Proposal", lines=10, interactive=False)

    # --- EVENT BINDINGS ---
    mode_radio.change(toggle_api_key_visibility, inputs=mode_radio, outputs=api_key_input)
    init_btn.click(
        launch_triad_system,
        inputs=[mode_radio, api_key_input],
        outputs=[splash_screen, main_dashboard, active_mode, active_key, system_status_bar],
    )
    submit_btn.click(
        execute_triad_query,
        inputs=[user_input, active_mode, active_key],
        outputs=[system_status_bar, final_output, vote_output, ego_thought,
                 id_prop, ego_prop, superego_prop, crit_output, debate_file],
    )

if __name__ == "__main__":
    # Gradio 6: theme and css belong to launch(), not to gr.Blocks()
    demo.launch(theme=gr.themes.Monochrome(), css=CSS, inbrowser=True)