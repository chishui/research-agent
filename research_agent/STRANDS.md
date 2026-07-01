# Strands SDK port

This is the [Strands Agents SDK](https://strandsagents.com) version of the
research agent. It does the same job as the hand-rolled `agent.py` but lets the
Strands `Agent` drive the tool-use loop instead of the manual loop in
`agent.py`.

## What changed

| Concern | Hand-rolled (`agent.py` / `tools.py`) | Strands port (`strands_agent.py` / `strands_tools.py`) |
|---|---|---|
| Loop | manual `while` over `messages.create(...)` | `strands.Agent.__call__` runs the loop |
| Model | `AnthropicBedrock` client directly | `strands.models.BedrockModel` |
| Tools | `Tool` subclasses + JSON `input_schema` | `@tool`-decorated methods (schema inferred from signature + docstring) |
| Shared state | `ToolContext` dataclass | `ResearchTools` instance attributes (class-based-tools pattern) |
| System prompt | `SYSTEM_PROMPT` in `agent.py` | **imported from `agent.py`** — single source of truth |

The research logic itself (`research.py`, `common/`, `providers/`) is **unchanged**
and shared by both versions. Tool names are identical, so `SYSTEM_PROMPT` is
valid for both.

## Why a class for the tools

The workflow passes papers between tool calls by short handle (`p1`, `p2`, …) to
keep large paper bodies out of the model's context. Strands' built-in
`agent.state` only stores JSON-serializable values, but our handles map to live
`Paper` / `ParsedPaper` / `PaperSummary` objects. So we use the Strands
[class-based tools](https://strandsagents.com/docs/user-guide/concepts/tools/custom-tools/)
pattern: `ResearchTools` holds the registries as instance attributes and every
`@tool` method reads/writes the same `self`, exactly as `ToolContext` did. A
fresh `ResearchTools` is created per `run()` so runs don't share state.

## Prerequisites

The Strands SDK requires **Python 3.10+**. The original project venv
(`../bin/python`) is **3.9**, so this port uses a separate environment. Python
3.12 is installed via pyenv (`pyenv local 3.12.13` pins it in this directory),
and the dependencies live in a dedicated `.venv-strands`:

```bash
python -m venv .venv-strands          # resolves to 3.12 via the pyenv pin
source .venv-strands/bin/activate
pip install -r requirements-strands.txt   # strands-agents
# plus the existing pipeline deps used by common/ + providers/:
pip install anthropic beautifulsoup4 requests docling
```

`docling` is only needed for the PDF-parse path (`fetch_paper` / `pdf_parse`); it
pulls in torch and is a large install.

Bedrock auth is the same as the existing agent. Two credential paths are used:
`BedrockModel` (the Strands agent loop) resolves standard AWS credentials /
region, while the research tools' inner `AnthropicBedrock` calls use
`AWS_BEARER_TOKEN_BEDROCK`. To pin a region, pass `region_name=` to
`StrandsResearchAgent` (or set `AWS_REGION`).

## Run

```bash
.venv-strands/bin/python strands_agent.py "how much protein should I take per day?" -v
```

Verified end-to-end on `us.anthropic.claude-sonnet-4-6` in `us-west-2`: the
Strands loop runs scholar_search → check_relevance → fetch_paper/fetch_url →
summarize_paper → aggregate_summaries and emits the report structure defined in
`SYSTEM_PROMPT`.

Or from Python:

```python
from strands_agent import StrandsResearchAgent
print(StrandsResearchAgent().run("effect of intermittent fasting on health"))
```

The original `agent.py` / `main.py` still work unchanged as the legacy
(no-framework) implementation.
