# Better Compact for OpenCode V2

Provider-visible context reduction that keeps current instructions, task trajectory,
native tool pairs and encrypted provider state intact. This package requires
OpenCode 2.0.18 or newer; it has **no V1 plugin entrypoint or compatibility layer**.

## Install

Add the published package to `plugins` in `opencode.jsonc` or load a local
package directory. OpenCode V2 automatically loads the package's `./tui`
extension in the terminal:

```jsonc
{
  "$schema": "https://opencode.ai/config.json",
  "plugins": ["better-compact@0.3.0"]
}
```

For a local checkout, build first and point `plugins` to the package **directory**,
not a `.ts` file. Configure the optional automatic fallback separately with
OpenCode's native `compaction.auto` setting. The plugin does not silently change
that setting or hijack the built-in `/compact` command.

## Behavior

- Before an agent model request (including assistant/tool continuations), cheap
  pruning acts on V2 AI messages only. An unsafe, expanding or unpaired output is
  discarded; the persisted transcript is never edited by the request hook.
- Exact native message deltas are written privately under the session's project
  `.opencode/better-compact/v2/sessions/<session>/archives/` and indexed by an
  ownership-checked catalog. Older raw archives expire after seven days; IDs
  remain as tombstones. The catalog is ignored by Git.
- If cheap pruning cannot approach the configured target, a bounded summary
  model can produce a validated, trajectory-aware handoff plus a separate
  ≤100-word archive description. It receives full human wording and bounded
  tool evidence, not opaque encrypted state. Failed or underspecified output
  cannot replace the request and is not repeatedly billed on stable replay.
- Human wording remains live at the first boundary. An older cohort may retire
  only after a later validated handoff includes its exact current wording and
  its own session archive is verified. Encrypted reasoning/checkpoints are
  never replaced by an ordinary text summary.
- The `better_compact_recall` tool exposes a bounded, explicit catalog, excerpt
  or exact UTF-8 page. It is tool-permission gated and refuses foreign sessions,
  changed files, expired archives and arbitrary model-supplied paths. Recalled
  history is untrusted evidence, not a new user instruction.

## Commands

`/better-compact` queues a V2 session checkpoint at the next safe point. The
server compaction hook validates the summary before submitting it; models using
OpenCode's native encrypted compaction keep their native path. The built-in
`/compact` stays available. `/better-compact-status` reports exact and described
archive counts and checkpoint state through the server RPC; neither command
creates a hidden user prompt.

## Configuration

The plugin reads global `~/.config/opencode/better-compact.jsonc`, optional
`$OPENCODE_CONFIG_DIR/better-compact.jsonc`, then the nearest project's
`.opencode/better-compact.jsonc`. `.json` works where `.jsonc` is absent.
Overrides inherit by field, including exact provider/model IDs.

```jsonc
{
  "$schema": "https://raw.githubusercontent.com/AshishKumar4/better-compact/main/packages/opencode/better-compact.schema.json",
  "enabled": true,
  "compaction": {
    "automatic": true,
    "preset": "custom",
    "summaryModel": "openai/gpt-6-luna",
    "summaryEffort": "high",
    "custom": {
      "triggerPercent": 66,
      "targetPercent": 25,
      "recentToolTokens": 15000,
      "recentReasoningTokens": 28000,
      "prefixSummary": true,
      "collapsePercent": 45
    }
  }
}
```

The target is **best effort**, not permission to truncate active instructions.
The model's actual `limit.context` and last provider usage determine the
trigger, with a hard-overflow guard. A summary model and its effort variant
must be available for model-backed handoffs; unavailable models leave cheap
pruning or the original request intact.

This fork retains the upstream AGPL-3.0-or-later license.
