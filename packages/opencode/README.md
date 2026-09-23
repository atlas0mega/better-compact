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

| Command                    | Action                 |
| -------------------------- | ---------------------- |
| `/better-compact`          | Run Better Compact now |
| `/better-compact context`  | Show context usage     |
| `/better-compact stats`    | Show the active plan   |
| `/better-compact help`     | List commands          |
| `/better-compact-settings` | Open settings          |

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

For OpenCode scratch summaries, selected assistant turns are spread across up to five
model calls per compaction. The call count scales with estimated input size; each call
targets 12k input tokens and is capped at approximately 24k and 16 turns. Jobs beyond
that budget keep the deterministic summary and transcript reference. The calls can run
in parallel up to `compaction.custom.summarizerConcurrency`. Responses contain one
validated summary per turn, with a batch-wide output target that grows sublinearly
with the turn count. Three failed calls in a row pause further summary calls for five
minutes; that failure cooldown is separate from the per-compaction five-call limit.
If the validated batch would make the active context larger than the
deterministic plan, Better Compact retains the smaller plan. Original turns
remain in the citable transcript, and replay keeps the same transformed
prefix for prompt caching; the target remains best-effort.

The first last-resort prefix also needs synthesis: simply listing hundreds of
assistant-turn previews can itself exceed the target. Better Compact splits that
chronological progress into up to five roughly equal-sized, complete chunks,
preferring about 23k estimated input tokens per call. When five calls at that
size cannot cover the entire history, the chunks grow evenly up to the
configured summary model's context limit, with headroom reserved for instructions
and output. The calls run concurrently and assemble the six task-state sections
in original order.
Each chunk retains its transcript reference; original user instructions are
inserted verbatim from the source turns, never rewritten by the summarizer.
The plan is stored only if **all** chunks validate and the assembled prefix is
smaller. If the history cannot fit in five calls within the summary model's
context window, its context limit is unavailable, or a response fails, the
deterministic prefix stays in place—no historical slice is silently omitted.
Older cached deterministic prefixes get one upgrade attempt per unchanged range
and summary model, including when a previous attempt used the fixed-size chunks,
including below the trigger; accepted plans replay byte-stably for caching.

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
