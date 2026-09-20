# Rutgers CS Course Advisor

A research framework for course recommendations, prerequisite checks, course
information, and transcript-informed advising. The parser identifies intent,
the orchestrator routes work to specialized agents, and their results inform
the final response.

**Current entry point: `driver3.py`. Current agents: `agents2/`.**

## Quick start

Run from the repository root using Python 3.11 or newer:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If you already use the local `myenv` environment, you can keep using it.
For a new checkout, copy `.env.example` to `.env`. Preserve your existing
`.env` if it already contains credentials.

```dotenv
AZURE_OPENAI_ENDPOINT="https://course-planner-ai-resource.services.ai.azure.com/openai/v1"
AZURE_OPENAI_API_KEY="your-api-key"
AZURE_OPENAI_EMBEDDING_DEPLOYMENT="text-embedding-3-small"
# Example only: these must be actual deployment names on your Azure resource.
AZURE_OPENAI_PARSER_DEPLOYMENT="gpt-5-mini"
AZURE_OPENAI_ORCHESTRATOR_DEPLOYMENT="gpt-5"
AZURE_OPENAI_DATA_DEPLOYMENT="gpt-5-mini"
AZURE_OPENAI_CONSTRAINT_DEPLOYMENT="gpt-5-mini"
AZURE_OPENAI_PLANNING_DEPLOYMENT="gpt-5-mini"
AZURE_OPENAI_TRANSCRIPT_DEPLOYMENT="gpt-5-mini"
AZURE_OPENAI_RESPONSE_WRITER_DEPLOYMENT="gpt-5-mini"
```

All deployment values must match deployments on your Azure resource. Semantic
search requires a separate embedding deployment on that resource. The endpoint
must end in `/openai/v1`; the resource's `.openai.azure.com/openai/v1` URL is also
configurable. Authentication uses the API key in `.env`.

All seven agent variables are required: absent or blank values fail at startup
with the missing variable's name. Model selection flows from `.env` through
`get_azure_openai_settings().models` (`ModelConfig`) into `build_workflow`, then
each agent's constructor. The router and fallback response writer have separate
assignments. Change any of these values and restart to run another experiment;
no Python edits are needed. Existing process environment variables take precedence
over `.env`, following the existing dotenv behavior.

The configuration above is only an example, not hard-coded model policy. For
all-mini experiments set all seven assignments to your mini deployment; to test
a larger planning or router model, change only those assignments. Models must
support the Responses API and capabilities used by that agent (including vision
for transcript images). Deployment existence/compatibility is checked by Azure
when called, not by these local configuration checks.

There is no global deployment setting. Startup DAG parsing uses the parser
deployment; the shared client defaults to the orchestrator deployment, while each
agent explicitly supplies its own assignment. Embeddings
remain independently configured. Endpoint, authentication, STAR edges, prompts,
guardrails and evaluation behavior are unchanged.

```powershell
python driver3.py
```

Ask for recommendations or prerequisite information, or provide a transcript PDF
path when prompted. Enter `quit` to exit. Relative PDF paths resolve from your
terminal's working directory; absolute paths also work.

## Undergraduate and graduate advising

The advisor keeps undergraduate and graduate students in separate course pools.
It asks for academic level when unknown and remembers the answer for the session.
For example: "I'm a first-year CS master's student interested in machine learning"
or "I'm an undergraduate junior interested in systems." Asking about a graduate
course does not change an undergraduate profile. Explicit profile corrections clear
prior course references so follow-up questions cannot reuse the other pool.

`rutgers_courses.json` remains the undergraduate catalog. The separate
`rutgers_graduate_courses.json` contains 73 graduate course/registration records
from the [department course index](https://www.cs.rutgers.edu/academics/graduate/course-synopses),
checked September 20, 2026. Each record has credits (including variable ranges),
source URL, verification status, program scope, and a recommendation flag.
There are 46 records in the ordinary graduate recommendation pool. Research,
independent study, proseminar, administrative and MSDS capstone registrations,
CS-degree-excluded courses, and incomplete/historical records remain available
for same-level lookup but are excluded from ordinary recommendations. Historical
details are explicitly labeled; CS 503's conflicting credits remain unknown.
Special-topic titles describe a catalog course, not a verified section topic.

`academic_level` is `undergraduate` or `graduate`; `graduate_program` is `masters`,
`phd`, or `msds` when explicitly known. Graduate transcripts are not classified by
undergraduate earned-credit thresholds. Retrieval, exact lookup, candidate reuse,
constraints, and ranking enforce the level boundary. Live schedule requests and
caches distinguish U/G offerings. Unavailable or empty schedules leave offering
status unknown instead of claiming that catalog entries are scheduled.

Graduate prerequisites retain their source text, with structured rules for the
unambiguous course-code relationships. Transcript checks report known matches
and gaps, but overall graduate eligibility remains `requires_verification`:
admission background, equivalents, permission, and program rules are not fully
modeled. Graduate pathway requests return these requirements and an explicit
unresolved result; they do not traverse the undergraduate DAG or manufacture an
undergraduate enrollment plan. The existing BFS/DFS pathway feature remains for
undergraduate advising.

A separate graduate visualization is generated automatically at startup:
`agents2/graduate_prereq_graph.png`, with its structured records in
`agents2/graduate_prereq_dag.json`. It includes all 73 catalog entries and draws
only the 13 recorded graduate prerequisite relationships. Solid/dashed arrows
distinguish required courses from alternatives; program-specific edges are labeled.
Entries without recorded edges do not imply an absence of prerequisites. This
graph is explanatory and does not enable automatic graduate pathway eligibility.
It rebuilds when its source data changes. Regenerate it without Azure or SOC:

```powershell
python -m agents2.graduate_graph
```

The embedding collection name includes the catalog fingerprint and a schema
version. First startup after this change builds a new index using the configured
Azure embedding deployment; old collections are preserved. Future catalog edits
create a fresh collection automatically. Offline tests do not call Azure or SOC.
The per-turn JSONL metrics include `academic_profile` for separating research
cohorts without changing existing CSV columns. This does not implement LinUCB.

## Repository layout

| Path | Purpose |
| --- | --- |
| `driver3.py` | Interactive workflow and feedback collection |
| `agents2/` | Parser, orchestrator, data, constraint, planning, and transcript agents |
| `agents2/paths.py` | Locations of current resources and outputs |
| `agents2/query_schema.json` | Query interpretation schema |
| `agents2/prereq_dag.json` | Cached prerequisite graph |
| `rutgers_courses.json` | Undergraduate course catalog |
| `rutgers_graduate_courses.json` | Graduate catalog with provenance and verification flags |
| `agents2/academic_profile.py` | Academic-level boundaries and graduate prerequisite checks |
| `query_logger3.py` | CSV logging and schema handling |
| `query_log3.csv` | Current accumulated research results |
| `chroma_db/` | Existing persistent semantic-search index |
| `analysis/scripts/` | Research analysis scripts |
| `analysis/figures/` | Generated comparison figures |
| `data_collection/` | Course and instructor data collection utilities |
| `archive/` | Earlier implementations, experiment logs, and caches |
| `tests/` | Offline configuration, workflow, and logging checks |

The current workflow resolves its catalog, schema, DAG, vector index, `.env`,
and default query log relative to the repository, even when launched from
elsewhere. The graph image remains at `agents2/prereq_graph.png`.

## Logging and research outputs

### Prerequisite pathways

The prerequisite route now runs deterministic search in
`agents2/pathway_search.py`, alongside course lookup. Try:

- "How can I take Distributed Systems given my transcript?"
- "What is the shortest path to Machine Learning Principles?"
- "Show me alternative ways to reach that course."
- "Which eligible courses could I explore while working toward it?"

BFS searches completed-course states for the fewest additional courses. DFS
explores alternative prerequisite choices. Both respect every AND requirement
and at least one choice in each OR group, with a limit of 5,000 states per search
and up to four plans. Shared prerequisites count once. Plans group courses into
parallel prerequisite stages and identify the fewest stages among returned plans;
that is different from the fewest courses and is not a verified semester schedule.

Completed/transfer/AP credit reduces the remaining plan; in-progress courses are
explicitly conditional on passing. Without a transcript, plans start from no
earned credit. Eligibility here is prerequisite-based, not enrollment approval:
offerings, credit limits, minimum grades, and corequisites are not fully modeled.

Missing external catalog nodes and instructor permission are never assumed
satisfied. If a complete route cannot be established, conditional plans show the
known CS portion and external requirements to verify. Search limits and cycles
are reported. Existing prerequisite checks also withhold eligibility for unknown
courses or unverified instructor permission.

Pathway turns retain the existing `data_prereq` logging step and include
`prereq_dag.json`, `BFS`, and `DFS` in the data source list.

Qualifying recommendation/information turns append to `query_log3.csv`, retaining
existing rows. Logs include response time, routing steps, agents and sources,
token usage, satisfaction, and hallucination feedback. The current driver skips
transcript uploads and clarification turns in its normal research logging path;
token-limit failures have a separate logging path.

Generate current multi-agent figures or the archived single-agent comparison:

```powershell
python -m pip install -r analysis/requirements.txt
python analysis/scripts/analysis3.py
python analysis/scripts/single_analysis.py
```

These scripts write to `analysis/figures/`. Historical analysis scripts and
iteration-specific outputs remain under `analysis/` for reference.

## Session conversational memory

`ConversationState` separates the recent raw message window, durable structured
session facts, and resettable turn/routing state. The existing parser call emits
optional `memory_updates` for explicit user interests, career goals and preferences.
Lists support case-insensitive deduplication, addition, removal and replacement;
preferences overwrite by key or clear with null. Changing from challenging courses
to a lighter workload updates the same difficulty preference. Structured memory
is not extracted from assistant recommendations or used as transcript evidence.

After updates, omitted current-query entities inherit session interests/goals and
preferences. Data retrieval and planning therefore receive these facts even when
the user does not repeat them. Current explicit query entities take precedence.
The parser receives at most four recent messages (bounded excerpts), plus compact
structured memory, instead of an ever-growing parser thread. The application still
retains its existing 12-message window. No extra memory LLM call is made.

Successful planning, lookup and prerequisite results update separate ordered
`last_recommendations`, `last_lookup_courses`, and `last_pathway_targets` lists,
bounded to seven references each. The parser emits `course_reference` to select
the appropriate set. “Those” normally selects the latest set; “the second one”
selects its second entry. “Which two should I take?” references the whole recent
recommendation set for reranking, not automatically its first two entries.
Recommendation follow-ups retrieve those exact codes from the catalog and retain
existing transcript/availability filtering. An invalid reference requests clarification
instead of silently substituting all courses ever discussed.

`get_context(role)` provides copy-safe inspection/projection: parser gets bounded
recent messages and structured memory; router and response writer get structured
facts/references through routing context; planning gets structured facts/references
alongside candidates and constraints. Data uses enriched retrieval entities;
constraints keep their existing transcript/course inputs without unrelated chat.
No memory is printed automatically or added to research logs.

Interests, goals, preferences, recent result references, transcripts, resolved courses,
semester and messages survive turn resets. Routing results/events/iterations and
query usage reset; session usage persists. No rolling summary or restart persistence
was added: structured facts provide bounded memory without another model call,
and restarting still starts a new conversation. Research logs are not memory storage.

Limits: natural-language extraction/reference selection still depends on the parser;
offline tests verify application behavior with controlled parser outputs, not live
model accuracy. Recent recommendations reflect validated planning output, not a
reparse of the final prose answer. Existing hybrid-intent and multi-course eligibility
limitations remain; remembered references do not alter those guardrails or DAG logic.

### Pending clarification tasks

Advising is scoped exclusively to the Rutgers–New Brunswick dataset. Campus is
not a query entity, preference or clarification requirement. Parser outputs are
filtered at the scope boundary, and campus-only router clarifications cannot block
a known course. The fixed `campus=NB` parameter remains solely in the Rutgers
schedule API request to retrieve this application's schedule; it is not user input
or shared conversational state.

When the router emits `clarify`, `ConversationState.pending_clarification` retains
the original query/intent, known entities, requested missing fields and question.
Parser reference-clarification prompts also create pending state. This survives
turn resets but is separate from permanent interests and preferences.

On the next turn, the same parser call classifies the reply as `resume`, `incomplete`
or an explicit `new_task`. Application code (not a second reclassification call)
merges nonempty reply entities over known values and preserves the original intent.
Empty/null extraction fields do not erase known facts. Missing fields are satisfied
incrementally by supplied reply entities, before session defaults are applied.
The original question is passed to the router alongside merged entities, preserving
personal-eligibility wording. Research logs still record the actual new user message.

Thus “How can I take machine learning?” → “Which course?” → “Machine Learning
Principles” resumes prerequisite/pathway handling. “Actually, just tell me what
it covers” can cancel the pending task and start course-info handling instead.
Completed merges clear pending state before routing resumes; incomplete replies
retain it and ask for remaining fields. If the router needs another clarification,
it creates a new pending task with the merged context. No topology/deployment,
ranking, DAG or pricing changes are involved.

For debugging, inspect `state.pending_clarification` and the per-turn
`state.clarification_events` (original intent, missing fields, merge/clear/change
flags). They are not automatically printed or written into cost logs. Natural-language
task-change detection and ambiguous answers still depend on parser output. Router
clarifications should supply `missing_fields`; omitted fields use an intent-based
fallback (target course for course-info/prerequisite tasks, otherwise interests).
Questions emitted as ordinary `respond` text are not automatically recognized as
clarifications. Restart persistence remains intentionally unsupported.

## Inference-cost research metrics

`driver3.py` records model-returned usage for parser, router, response writer,
planning, constraint, transcript, query-rewrite and embedding calls. Costs are
local estimates named `estimated_inference_cost_usd`, not Azure invoice totals.
No Cost Management request is made. No token counts are inferred from text.

Configure `pricing.json` using the shape in `pricing.example.json`. Map the exact
Azure deployment name to verified model/version, deployment type/region, USD
input/output rates per million tokens, optional cached-input rate, and pricing
source/effective date. Add a separate entry for the embedding deployment if used.
The checked-in configuration is intentionally empty: unknown, incomplete or invalid
pricing produces **unavailable**, never an assumed free or guessed-price call.
Example rates in tests are synthetic, not suggested Azure prices.

Cached inputs are a subset of input tokens and receive the configured cache rate
instead of being charged twice. Missing cache counts stay null; without exposed
cache usage or a configured cache rate, regular input pricing applies. A total
containing any unpriced call is unavailable, while its token counts remain logged.

For an existing legacy `query_log3.csv`, evaluated turns append in its original
schema, and richer rows also append to `query_log3_metrics.csv`. New empty CSVs
use the richer schema directly. Historical rows are never migrated or rewritten.
`query_log3_calls.jsonl` contains per-call costs, usage, pricing provenance, parsed
intent, proposed/executed routing decisions and intervention reasons, joined to
the richer CSV by `query_id`. It records uploads, clarification and failed turns
as well as evaluated turns, without copying prompts or transcript contents.
For new query rows, `model_id` lists the distinct deployments actually called,
separated by `|` (including embeddings), rather than implying a single model.
JSONL records the full role-to-deployment `model_config`, including roles not
called that turn, separately from actual per-call deployment IDs. Each call is
priced using its own deployment entry in `pricing.json`; configure every model
used in the experiment or its cost remains unavailable. Existing CSV headers
and historical rows are preserved.

Query counters reset each turn; session counters persist. Startup DAG/index calls
are recorded separately with `phase="startup"` and included in session totals,
not the first query. `llm_call_count` excludes embeddings; `model_call_count` and
total input/output usage include them. JSONL also exposes LLM-only token totals.
Deterministic BFS/DFS, cache reads and HTTP catalog requests are not LLM calls.
These comprehensive counters should not be treated as directly equivalent to
historical rows that only captured usage from some agents.
The console prints a separate compact research cost summary after evaluation.

Coverage is the current non-streaming model calls inside the workflow (workflow
event streaming is not model-token streaming). Failed requests that return no
usage, SDK retries without a returned response, external processes and standalone
DAG runs are not billable-usage measurements. These estimates exclude hosting,
storage, provisioned capacity, discounts and other billing details. Review the
metered response-parser hook when upgrading Agent Framework; offline tests cover
the installed SDK boundary but do not validate Azure billing or live integration.

### STAR baseline boundaries

Only `build_workflow(..., topology="star")` is supported. Logical capabilities
remain separate from their physical executors: `data_fetch`, `data_lookup` and
`data_prereq` share DataExecutor; both constraint capabilities share
ConstraintExecutor. Existing `plan_steps` records logical dispatches and
`agents_invoked` records physical participation. Routing sidecars distinguish
the proposed route from dispatch and record guardrail interventions. A dispatch
does not itself prove successful completion of the capability.

The LLM router, sequential hub/spoke edges, terminal transcript spoke and current
intent/dependency restrictions are unchanged. A hybrid request such as “Recommend
AI courses I can take now, and show the shortest prerequisite route for the rest”
can still be restricted to recommendation capabilities by its primary intent;
this limitation is tested rather than silently relaxed. Future topology work
should revisit hybrid-capability policy, dependency scheduling, concurrency-safe
aggregation and equivalent experimental workloads. BFS/DFS remains isolated
domain logic, not a topology mechanism.

## Offline validation

```powershell
python -m unittest discover -s tests -v
```

Tests use mocks and temporary files: they do not call Azure, modify the live
vector index, or append test rows to the research log.

## Historical work

See [archive/README.md](archive/README.md) for the older experiments. Their files
and results are preserved; the maintained entry point is `driver3.py`.
Existing credentials, transcript files, and the active vector index remain in
place. Ignore rules do not remove files already tracked by Git.

## Contributors

Jelani Beaugris, Arpita Biswas — framework development and testing.
