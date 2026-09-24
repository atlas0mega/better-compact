# Better Compact for OpenCode

## Compaction reasoning per provider/model

Set `summaryEffort` inside `compaction`, a `compaction.providers.<provider>` entry,
or that provider's `models.<model>` entry. Missing values inherit model →
provider → global, independently of thresholds and other settings.

For example, set a provider's `summaryEffort` to `"high"` and one model's to
`"low"` to use cheaper reasoning only for that model's compaction summaries.
Accepted values are `inherit`, `low`, `medium`, `high`, `max`, and `off`.
`inherit` keeps the active conversation variant; `off` disables LLM summaries
(it does not mean reasoning-disabled summaries). `max` selects `max`, or `xhigh`
when that is what the model advertises. Unsupported variants retain the active
variant. This applies to automatic and manual summaries; an explicit manual
reasoning selection takes precedence. Main-conversation reasoning is unchanged.

## Provider/model compaction overrides (fork)

In `better-compact.jsonc`, settings inherit **field by field**: model override,
then provider override, then global compaction settings. Unspecified fields keep
their inherited values. IDs are exact OpenCode provider/model IDs, not wildcards.

```jsonc
{
    "compaction": {
        "preset": "custom",
        "custom": { "triggerPercent": 85, "targetPercent": 35 },
        "providers": {
            "runpod": {
                "custom": { "summarizerConcurrency": 3 },
                "models": {
                    "qwen3.8-27b-ud-q6-k-m": { "triggerTokens": 181000 },
                },
            },
        },
    },
}
```

`triggerTokens` and `targetTokens` are optional positive safe integers that
override percentage budgets, including with named presets. `null` clears an
inherited absolute budget and restores percentage behavior. To replace an
inherited token budget with a percentage, clear that token field explicitly.
Partial `custom` settings and scoped `automatic`, `preset`, and `summaryEffort`
inherit normally. Global/config-directory/project layers deep-merge scoped
entries. The global-settings UI preserves these JSONC entries; edit scoped
overrides in the file.

These settings belong to Better Compact, not an unsupported OpenCode provider
`compaction` property. Keep the model's real per-request `limit.context`
accurate. Token budgets do not expand that window. Triggering starts planning;
turn boundaries, estimates, protected history and pruning eligibility still
affect when compaction occurs and whether it reaches the target.
OpenCode normally keeps the last two real user turns raw. If the full span
between them is already larger than the target (as in a long agent/tool loop),
it keeps the most recent user turn raw instead. Older **whole turns** become
eligible for pruning, remain in the lossless transcript, and their user
instructions stay live or carry through the prefix summary.
The choice is recorded in the plan and replays with a stable prompt prefix.

This fork retains the upstream AGPL-3.0-or-later license.

<p align="center">
  <img src="https://raw.githubusercontent.com/AshishKumar4/Better-Compact/main/assets/readme/hero.svg" alt="Better Compact staged context pruning." width="100%">
</p>

<p align="center">
  <a href="https://www.npmjs.com/package/better-compact"><img src="https://img.shields.io/npm/v/better-compact?style=flat-square" alt="npm version"></a>
  <a href="https://www.npmjs.com/package/better-compact"><img src="https://img.shields.io/npm/dm/better-compact?style=flat-square" alt="monthly downloads"></a>
  <a href="https://github.com/AshishKumar4/Better-Compact/actions/workflows/ci.yml"><img src="https://img.shields.io/github/actions/workflow/status/AshishKumar4/Better-Compact/ci.yml?branch=main&style=flat-square&label=CI" alt="CI status"></a>
</p>

<p align="center"><sub>Edited and maintained by Claude. Provided as-is.</sub></p>

Staged context pruning for OpenCode 1.17.13 and newer.

## Install

```bash
opencode plugin better-compact --global
```

Restart OpenCode after installation.

The installer updates:

```text
~/.config/opencode/opencode.json
~/.config/opencode/tui.json
```

JSONC files are preserved when present.

## Commands

| Command                    | Action                                                          |
| -------------------------- | --------------------------------------------------------------- |
| `/better-compact`          | Run now if idle; otherwise queue once after the active turn     |
| `/better-compact context`  | Show context usage                                              |
| `/better-compact stats`    | Show the active plan                                            |
| `/better-compact help`     | List commands                                                   |
| `/better-compact-settings` | Open settings                                                   |
| `/compact`, `/summarize`   | TUI aliases for `/better-compact` when its TUI plugin is loaded |

When requested during an active assistant/tool loop, manual compaction waits for the
next completed assistant/tool step and finishes before the following provider request;
idle handles a completed terminal turn. The command's
no-reply messages keep the session's current agent, model, and variant; a separate
summary-model effort does not change the conversation variant.
The TUI plugin shadows OpenCode's native `session.compact` command at higher keymap
priority, including its `/summarize` alias. This redirect does not disable native
compaction invoked directly through OpenCode's API. Restart a running TUI to load
plugin code changes.

## Configuration

Better Compact reads these files in order:

1. `~/.config/opencode/better-compact.jsonc` or `.json`
2. `$OPENCODE_CONFIG_DIR/better-compact.jsonc` or `.json`
3. `.opencode/better-compact.jsonc` or `.json`

Project settings override global settings.

```jsonc
{
    "$schema": "https://raw.githubusercontent.com/AshishKumar4/better-compact/main/packages/opencode/better-compact.schema.json",
    "enabled": true,
    "compaction": {
        "automatic": true,
        "preset": "light",
        "summaryEffort": "inherit",
    },
}
```

| Preset     | Trigger | Target | Recent tool budget |
| ---------- | ------: | -----: | -----------------: |
| `light`    |     85% |    35% |         40k tokens |
| `moderate` |     75% |    25% |         30k tokens |
| `max`      |     60% |    15% |         12k tokens |

Changes made in `/better-compact-settings` apply to later runs without restarting OpenCode.

To run scratch summaries on another model without changing the conversation model or its
compaction thresholds, set `compaction.summaryModel` to a `provider/model-id` string:

```jsonc
{
    "compaction": {
        "summaryModel": "openai/gpt-6-luna",
        "summaryEffort": "high",
    },
}
```

The setting inherits through provider/model compaction overrides. Set a scoped `summaryModel`
to `null` to use the active chat model for that scope. When switching models,
`summaryEffort: "inherit"` uses the summary model's default variant rather than carrying
over the conversation model's variant.

The TUI manual command uses `noReply: false` when the session is busy, joining
OpenCode's existing run; an idle invocation uses `noReply: true` and does not
start a chat turn. A busy manual request executes once after the assistant or
tool content advances, in the awaited pre-provider transform before the same
agent loop continues, or on `session.idle` when the turn terminates. An
unchanged queuing prompt alone does not activate it.

Automatic triggering uses the last completed provider response's reported
tokens, including in agent/tool loops; it does not infer a trigger from the
raw stored transcript. An outgoing request estimated beyond the model window
still uses the separate forced overflow guard. Planning prices the transformed
messages on one local scale; the previous provider reading is not subtracted
from a reconstructed history and added back as imaginary fixed overhead.
The report distinguishes local projected history from the provider reading,
which updates after the next response and may include unpriced system/tool
schemas and provider accounting.

In OpenCode's archive workflow, deterministic pruning comes first. The
25% target in the current live configuration is best-effort: a new last-resort
prefix starts only above 115% of the target. It retains the newest complete
pruned turns natively in available projected context, counting protected
reasoning, tools, the handoff, and raw tail together. OpenCode also
anchors its last five real assistant text outputs with the reasoning between
them, excluding tool calls. **Five is a reasoning-span anchor, not a cap on
assistant chat:** after reserving genuine user wording, additional whole
assistant responses can remain native up to the projected target before old
stubs and tool traffic consume the remaining headroom. Provenance-marked
Syndicate plugin injections are generated, tool-like traffic, not human turns
or user-first retention. Previously archived text remains available through
recall; projected tokens can differ from the next provider reading. A
provider-window buffer reserves room for the next response; if that reasoning
interval cannot fit, the outputs remain native and reasoning falls back to its
28k allowance, with the limit reported.
A previously applied prefix remains stable on replay. The
Luna/high **live handoff** runs synchronously before the next provider request
only when the resulting plan exceeds 115% of the target. It may use at most
six calls (up to five balanced concurrent source chunks plus one final handoff).
Archive descriptions are separate Luna/default jobs that run in the background;
the two paths share **at most seven calls per compaction**, including retries.
Each failed job may be attempted at most five times while slots remain.
An archive whose evidence fits the detected model window uses one live-handoff
call; larger work uses balanced chunks. ~23k estimated input tokens is a
preferred chunk size, not a hard ceiling. A chunk may grow to the summary model's detected
context limit minus a bounded output reserve (24k tokens on large models) and its
own prompt overhead. If exact archived payloads are too large, the summarizer
receives pruned evidence while the private archive retains the exact originals.
No historical chunk is silently dropped. A failed or non-reducing live handoff
retains the deterministic handoff and private archive for later recall. Once a
live handoff validates and makes the **complete** outgoing context smaller, it
replaces the covered old wording before the next provider call. Background
descriptions make pending archives ready without changing the active cached plan.
Stable plan replay makes no additional summary calls.
The session-scoped `better_compact_recall` tool supports `catalog`, `excerpt`, and
`page`. If the catalog marks `archivedSummaryAvailable`, an oversized or rejected
model response is available through `excerpt` or `page` with `artifact: "summary"`
and that archive ID. This historical output is not a validated live handoff;
both it and the exact native delta require read permission and bounded paging.
An unchanged manual command or a settings-only replan does not re-run the live
handoff on the same boundary. Pending descriptions are retried on plugin startup
for project catalogs under the cwd, with project-scoped SDK calls and a
cross-process per-session lock. Exact archive payloads and rejected-summary
sidecars expire after seven days on project activity or startup; catalog IDs
remain as tombstones. The latest complete older reasoning parts can be kept
under `compaction.custom.recentReasoningTokens` (base 28k in the user's setup,
with growth toward 25% of older reasoning limited by remaining target headroom
after other retained context); the
recent raw tail remains protected.
For long assistant/tool loops with no intervening user message, a target-sized
whole-turn tail may advance into the loop when the **complete outgoing plan**
is smaller than the ordinary user-anchored plan. The archived native turns
remain recoverable by session-scoped recall. Catalog writes are serialized
across processes so background descriptions cannot erase newly added deltas.

User-role prompts generated by the Syndicate Engine plugins carry a trailing
`[plugin-injection:<uuid>]` provenance marker. Better Compact recognizes this
marker as tool-like generated context: old prompts can be stubbed under the
tool-result retention budget, do not count as protected user instructions, and
are charged to the tool category in context reports. The current raw tail stays
intact; pruned prompt bodies remain available in the lossless reference
transcript. A prior plan containing these prompts is rebuilt once on the next
automatic request, even if it was already below the trigger. OpenCode's stored
message roles are not rewritten.

## Upgrade

Use an explicit version to force a fresh OpenCode package cache entry:

```bash
opencode plugin better-compact@0.2.9 --global
```

If the plugin is missing after a failed or stale install, clear its cache and restart OpenCode:

```bash
rm -rf ~/.cache/opencode/packages/better-compact*
```

## Uninstall

Remove `better-compact` from the `plugin` arrays in:

```text
~/.config/opencode/opencode.json
~/.config/opencode/tui.json
```

Restart OpenCode.

## Development

From the repository root:

```bash
pnpm install
pnpm --filter better-compact typecheck
pnpm --filter better-compact test
pnpm --filter better-compact build
pnpm --filter better-compact check:package
```

Use the checkout directly:

```json
{
    "plugin": ["file:///path/to/better-compact/packages/opencode/index.ts"]
}
```

Add the TUI entry to `~/.config/opencode/tui.json`:

```json
{
    "plugin": ["file:///path/to/better-compact/packages/opencode/tui.tsx"]
}
```

## Architecture

OpenCode runs Better Compact through two bundled entrypoints:

```text
dist/index.js   server hooks and pruning runtime
dist/tui.js     commands, settings, reports, progress UI
```

The server applies a virtual plan to each outgoing request. OpenCode session history remains unchanged.

The pruning order is:

1. Remove loaded skill text.
2. Supersede repeated reads and remove stale failed-tool inputs.
3. Replace old tool calls and results with action stubs.
4. Remove old reasoning if more space is needed.
5. Remove remaining old tool traffic if more space is needed.
6. Collapse selected assistant runs and summarize them.
7. Use a rolling prefix summary as a last resort.

Raw history for each planned range is stored under:

```text
.opencode/better-compact/sessions/<session-id>/<range-hash>.md
```

The model receives a reference to that file. Plans are range-hashed and replayed until prefix edits or context regrowth require a new plan.

## License

AGPL-3.0-or-later
