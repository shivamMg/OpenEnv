# Agent Sim Environment

`agent_sim_env` turns Agent Traces into a stateful OpenEnv simulation.
`SimGenerator` creates one task per source conversation, a semantic SQLite store, a
SQLite-backed tool caller, a global policy grader, and terminal database
graders. Runtime serving does not call an LLM.

Generated artifacts are deliberately server-only:

```text
data/tasks.json                 # tasks, expert actions, terminal grader code
data/store.db                   # immutable baseline relational state
server/tools.py                 # generated SQLite-backed tool caller
server/graders/policy.py        # generated global policy grader
```

The agent only sees the system prompt, tool definitions, visible messages, and
tool responses. Expert actions and grader code never appear in observations.

## Quick start

```bash
PYTHONPATH=src:envs uv run python -m agent_sim_env.server.app
```

Reset accepts an optional `task_id` or a seed. Each reset creates an immutable
initial database snapshot and a mutable per-episode SQLite copy. Generated tools
operate only on that mutable copy.

The action contract has two variants: `TextMessage` (`type: "text_message"`) and
`ToolCallAction` (`type: "tool_call"`). Rewards are:

| Grader | Weight | When it runs |
| --- | ---: | --- |
| `FinalStateCodeGrader` | 0.6 | Terminal step only |
| `ToolCallGrader` | 0.2 | Tool-call steps |
| `TextMessageGrader` (stemmed ROUGE-L) | 0.1 | Text-message steps |
| `PolicyGrader` | 0.1 | Every active step |

For an action $a_t$, the immediate reward is

$$
r_t = 0.2T_t + 0.1M_t + 0.1P_t + \mathbb{1}_{terminal}0.6F_t.
$$

The current unmatched expert action is graded first. If it scores zero, the
environment searches later unmatched actions of the same type and applies a
0.5 out-of-order shaping penalty. Threshold for Tool calls match is $0.8$ and for Text messages match is $0.6$. Episodes end after every expert action is matched or the
environment-level `MAX_STEPS` limit is reached.

## Generate artifacts

Install generator dependencies and set Azure OpenAI values:

```bash
export AZURE_OPENAI_ENDPOINT="https://<resource>.openai.azure.com/"
export AZURE_OPENAI_DEPLOYMENT="<deployment>"
export AZURE_OPENAI_API_KEY="<key>"
export AZURE_OPENAI_API_VERSION="2024-10-21"
uv run --extra generator python -m agent_sim_env.scripts.sim_generator \
	--traces-dir envs/agent_sim_env/traces \
	--output-dir envs/agent_sim_env
```

The generator validates generated Python, creates the database atomically, and
replays every observed trace tool call against a fresh SQLite copy before it
publishes the artifacts. Trace responses are validation targets, not a generic
call/response replay fixture.

Generated final-state and policy code execute in a constrained subprocess with
a wall timeout and best-effort POSIX resource limits. This is defense in depth
for generated, curated code—not a security boundary for hostile Python.
