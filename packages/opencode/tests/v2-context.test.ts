import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp, readFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import type { PluginConfig } from "../lib/v2/config"
import { loadV2Catalog, readV2Archive } from "../lib/v2/archive"
import { v2Codec } from "../lib/v2/codec"
import { compactV2Context, lastProviderTokens } from "../lib/v2/context"

const config: PluginConfig = {
    enabled: true, debug: false,
    commands: { enabled: true }, experimental: { allowSubAgents: false },
    compress: { permission: "allow" },
    compaction: { automatic: true, preset: "custom", summaryEffort: "high",
        custom: { triggerPercent: 66, targetPercent: 25, recentToolTokens: 15000,
            recentReasoningTokens: 28000, summarizerConcurrency: 7, prefixSummary: true,
            collapsePercent: 45 } },
}

test("V2 request seam archives originals, shrinks old tools, and keeps current intent", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-context-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const instruction = Message.make({ id: "instruction", role: "user", content: [Message.text("Never overwrite user files")] })
    const old = Array.from({ length: 12 }, (_, index) => [
        Message.make({ id: `call${index}`, role: "assistant", content: [{ type: "tool-call", id: `id${index}`, name: "read", input: { path: `src/${index}.ts` } }] }),
        Message.make({ id: `result${index}`, role: "tool", content: [{ type: "tool-result", id: `id${index}`, name: "read", result: { type: "text", value: "old file result ".repeat(210) } }] }),
    ]).flat()
    const current = Message.make({ id: "current", role: "user", content: [Message.text("Finish the active tests")] })
    const original = [instruction, ...old, current]
    const beforeTokens = v2Codec.estimateTurns(v2Codec.encode(original))
    const event = { sessionID: "ses_fixture", messages: [...original] }
    const input = { config, projectRoot: root, contextLimit: beforeTokens + 100 }
    const outcome = await compactV2Context(event, input)
    assert.equal(outcome.status, "applied")
    assert.ok((outcome.afterTokens ?? 0) < beforeTokens)
    assert.equal(event.messages[0]?.id, "instruction")
    assert.equal(event.messages.at(-1)?.id, "current")
    assert.deepEqual(original, [instruction, ...old, current], "the persisted source must remain untouched")
    const catalog = await loadV2Catalog(root, event.sessionID)
    assert.equal(catalog.entries.length, 1)
    const recovered = await readV2Archive(root, event.sessionID, catalog.entries[0]!.id)
    assert.equal(JSON.stringify(recovered), JSON.stringify(original.filter((message) => catalog.entries[0]!.messageIDs.includes(message.id!))))
    const pointer = event.messages.flatMap((message) => message.content)
        .filter((part) => part.type === "text").map((part) => part.text).join("\n")
    const path = pointer.match(/\.opencode\/better-compact\/v2\/sessions\/ses_fixture\/catalog\.json/)?.[0]
    assert.ok(path, "the model-visible reference must point to a real private catalog")
    assert.ok((await readFile(join(root, path), "utf8")).includes(catalog.entries[0]!.id))
    assert.doesNotMatch(await readFile(join(root, path), "utf8"), /old file result/)
    // The same native request creates no additional delta or duplicate model-visible text.
    const replay = { sessionID: event.sessionID, messages: [...original] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal((await loadV2Catalog(root, event.sessionID)).entries.length, 1)
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(event.messages))
})

test("V2 request seam compacts real ID-less tool messages without losing exact human wording", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-idless-context-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const original = [Message.user("Never overwrite user files"),
        Message.make({ role: "assistant", content: [{ type: "tool-call", id: "read-1", name: "read", input: { path: "file.txt" } }] }),
        Message.make({ role: "tool", content: [{ type: "tool-result", id: "read-1", name: "read",
            result: { type: "text", value: "old diagnostic ".repeat(4_000) } }] }),
        Message.assistant("The next test decision is to verify the original constraint."),
        Message.user("Continue with the next test without changing that constraint.")]
    const exactSource = JSON.stringify(original)
    const event = { sessionID: "ses_idless", messages: [...original] }
    const before = v2Codec.estimateTurns(v2Codec.encode(original))
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const input = { config: local, projectRoot: root, contextLimit: before + 100 }
    const outcome = await compactV2Context(event, input)
    assert.equal(outcome.status, "applied")
    assert.ok((outcome.afterTokens ?? before) < before * 0.2)
    assert.deepEqual(event.messages[0], original[0])
    assert.deepEqual(event.messages.at(-1), original.at(-1))
    assert.doesNotMatch(JSON.stringify(event.messages), /old diagnostic old diagnostic/)
    assert.equal(JSON.stringify(original), exactSource, "the persisted source cannot be changed by the request hook")
    const catalog = await loadV2Catalog(root, event.sessionID)
    const recovered = (await Promise.all(catalog.entries.map((entry) => readV2Archive(root, event.sessionID, entry.id)))).flat()
    assert.equal(JSON.stringify(recovered), JSON.stringify(original.slice(0, recovered.length)))
    assert.ok(recovered.every((message) => !message.id), "no virtual ID may enter exact archives")
    const replay = { sessionID: event.sessionID, messages: [...original] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(event.messages))
    assert.equal((await loadV2Catalog(root, event.sessionID)).entries.length, catalog.entries.length)
})

test("V2 seam leaves below-trigger requests untouched", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-context-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const message = Message.make({ id: "active", role: "user", content: [Message.text("Continue carefully")] })
    const event = { sessionID: "ses_small", messages: [message] }
    assert.equal((await compactV2Context(event, { config, projectRoot: root, contextLimit: 200000 })).status, "below-trigger")
    assert.deepEqual(event.messages, [message])
})

test("V2 provider usage includes cached and reasoning tokens; below-trigger usage does not invent local overhead", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-context-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const usage = lastProviderTokens([{ type: "assistant", tokens: {
        input: 100, output: 30, reasoning: 400, cache: { read: 9000, write: 300 },
    } }])
    assert.equal(usage, 9830)
    const event = { sessionID: "ses_provider", messages: [Message.make({ id: "now", role: "user", content: [Message.text("Keep working")] })] }
    assert.equal((await compactV2Context(event, { config, projectRoot: root, contextLimit: 10000,
        providerReportedTokens: 100 })).status, "below-trigger")
    assert.equal((await compactV2Context(event, { config, projectRoot: root, contextLimit: 10000,
        providerReportedTokens: usage })).status, "declined", "no old material exists for a plan")
})

test("V2 validated handoff shrinks older assistant trajectory and replays without a second Luna bill", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-handoff-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const messages = [
        Message.make({ id: "requirement", role: "user", content: [Message.text("Never overwrite user files")] }),
        ...Array.from({ length: 14 }, (_, index) => Message.make({ id: `progress${index}`, role: "assistant",
            content: [Message.text(`Implemented file ${index} and tested the path. ${"Old diagnostic detail ".repeat(160)}`)] })),
        Message.make({ id: "current", role: "user", content: [Message.text("Finish the active tests")], }),
    ]
    const before = v2Codec.estimateTurns(v2Codec.encode(messages))
    const handoff = ["## Decisions", "- Preserve existing files; the prior implementation paths passed initial checks.",
        "## Files & Symbols", "- src/feature.ts and tests/feature.test.ts were revised.",
        "## Errors (verbatim)", "- Old diagnostic detail was not needed for the latest tests.",
        "## What failed and why", "- Earlier attempts kept repeated diagnostic narration.",
        "## Constraints", "- Never overwrite user files.", "## Next step", "- Finish the active tests."].join("\n")
    let calls = 0
    const input = { projectRoot: root, contextLimit: before + 100,
        config, summary: { modelContextLimit: 100000,
            generate: async (_prompt: string, variant: string | undefined) => {
                calls++
                return variant ? JSON.stringify({ handoff }) : "Feature implementation, repeated diagnostics and the next test decision"
            } } }
    const first = { sessionID: "ses_handoff", messages: [...messages] }
    const outcome = await compactV2Context(first, input)
    assert.equal(outcome.status, "applied")
    assert.ok((outcome.afterTokens ?? before) < before)
    assert.equal(calls, 2, "one Luna/high handoff and one default-effort description")
    const catalog = await loadV2Catalog(root, "ses_handoff")
    assert.equal(catalog.checkpoint?.handoff, handoff)
    assert.equal(catalog.entries[0]?.status, "ready")
    assert.match(JSON.stringify(first.messages), /Never overwrite user files/)
    assert.match(JSON.stringify(first.messages), /Finish the active tests/)
    const replay = { sessionID: "ses_handoff", messages: [...messages] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(calls, 2, "stable replay must not spend another model call")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(first.messages))
})

test("V2 ID-less checkpoint stays visible and protects post-checkpoint work without another summary bill", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-continuity-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const constraint = Message.user("Never overwrite user files")
    const progress = Array.from({ length: 8 }, (_, index) => Message.assistant(
        `Implementation decision ${index}: diagnostics inspected and tests next. ${"Old routine output ".repeat(480)}`))
    const active = Message.user("Finish the current test decision")
    const initial = [constraint, ...progress, active]
    const handoff = ["## Decisions", "- Keep the original implementation decision and test trajectory.",
        "## Files & Symbols", "- src/feature.ts remains the active implementation.",
        "## Errors (verbatim)", "- No current regression confirmed.",
        "## What failed and why", "- The old routine output hid the current test decision.",
        "## Constraints", "- Never overwrite user files", "## Next step", "- Finish the current test decision"].join("\n")
    let calls = 0
    const summary = { modelContextLimit: 100000, generate: async (_prompt: string, variant: string | undefined) => {
        calls++
        return variant ? JSON.stringify({ handoff }) : "Validated implementation decisions, tests and preserved human constraints"
    } }
    const input = { config, projectRoot: root, contextLimit: v2Codec.estimateTurns(v2Codec.encode(initial)) + 100, summary }
    const first = { sessionID: "ses_continuity", messages: [...initial] }
    assert.equal((await compactV2Context(first, input)).status, "applied")
    assert.equal(calls, 2)
    assert.match(JSON.stringify(first.messages), /## Decisions/)
    const currentProgress = Message.assistant("The newest implementation test passed; now verify the next case.")
    const next = Message.user("Verify the next case without changing the prior constraint")
    const followUp = { sessionID: "ses_continuity", messages: [...initial, currentProgress, next] }
    const outcome = await compactV2Context(followUp, input)
    assert.equal(outcome.status, "applied")
    assert.equal(outcome.selectedStrategy, "checkpoint")
    assert.equal(calls, 2, "replay must not pay for a premature replacement summary")
    assert.match(JSON.stringify(followUp.messages), /## Decisions/)
    assert.ok(followUp.messages.some((message) => message === currentProgress), "new assistant work must remain exact")
    assert.ok(followUp.messages.some((message) => message === next), "the latest human correction must remain exact")
    assert.match(JSON.stringify(followUp.messages), /Never overwrite user files/)
    const catalog = await loadV2Catalog(root, followUp.sessionID)
    assert.equal(catalog.entries.filter((entry) => entry.status === "ready").length, 1)
    const archived = await readV2Archive(root, followUp.sessionID, catalog.checkpoint!.archiveID)
    assert.ok(archived.every((message) => !message.id), "the saved checkpoint must recover the original ID-less messages")
})

test("invalid V2 handoff cannot retire a constraint or bill again on an unchanged replay", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-invalid-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const messages = [Message.make({ id: "requirement", role: "user", content: [Message.text("Never overwrite user files")] }),
        ...Array.from({ length: 9 }, (_, index) => Message.make({ id: `work${index}`, role: "assistant",
            content: [Message.text("Implementation progress ".repeat(180))] })),
        Message.make({ id: "current", role: "user", content: [Message.text("Finish the active tests")] })]
    let calls = 0
    const input = { projectRoot: root, contextLimit: v2Codec.estimateTurns(v2Codec.encode(messages)) + 100,
        config, summary: { modelContextLimit: 100000,
            generate: async () => { calls++; return JSON.stringify({ handoff: "Everything is fine" }) } } }
    const first = { sessionID: "ses_invalid", messages: [...messages] }
    assert.equal((await compactV2Context(first, input)).status, "declined")
    assert.equal(calls, 1)
    assert.equal(JSON.stringify(first.messages), JSON.stringify(messages))
    assert.equal((await loadV2Catalog(root, "ses_invalid")).summaryAttempt?.reason, "invalid_output")
    const retry = { sessionID: "ses_invalid", messages: [...messages] }
    assert.equal((await compactV2Context(retry, input)).status, "declined")
    assert.equal(calls, 1)
    assert.equal(JSON.stringify(retry.messages), JSON.stringify(messages))
})

test("V2 round two retires only exact older ready wording after a validated replacement handoff", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-two-boundary-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const requirement = Message.make({ id: "requirement", role: "user", content: [Message.text("Never overwrite user files")] })
    const firstProgress = Array.from({ length: 12 }, (_, index) => Message.make({ id: `progress-${index}`,
        role: "assistant", content: [Message.text(`Implementation decision ${index}: ${"old diagnostic ".repeat(155)}`)] }))
    const correction = Message.make({ id: "correction", role: "user", content: [Message.text("Finish the active tests")] })
    const firstHistory = [requirement, ...firstProgress, correction]
    const secondProgress = Array.from({ length: 12 }, (_, index) => Message.make({ id: `later-${index}`,
        role: "assistant", content: [Message.text(`Follow-up decision ${index}: ${"new diagnostic ".repeat(155)}`)] }))
    const next = Message.make({ id: "next", role: "user", content: [Message.text("Investigate corrected regression")] })
    const secondHistory = [...firstHistory, ...secondProgress, next]
    const handoff = (latest: string) => ["## Decisions", "- Preserve user files; implementation and follow-up tests remain in progress.",
        "## Files & Symbols", "- src/feature.ts and tests/feature.test.ts.",
        "## Errors (verbatim)", "- The older diagnostic was resolved; the regression remains.",
        "## What failed and why", "- Repeated tool logs hid the relevant failure.",
        "## Constraints", "- Never overwrite user files.", "## Next step", `- Finish the active tests. ${latest}`].join("\n")
    let calls = 0
    const summary = { modelContextLimit: 100000,
        generate: async (_prompt: string, variant: string | undefined) => {
            calls++
            return variant ? JSON.stringify({ handoff: handoff(calls <= 2 ? "" : "Investigate corrected regression.") })
                : "Implementation tests, historical diagnostics and corrected regression lookup"
        } }
    const initial = { sessionID: "ses_rounds", messages: [...firstHistory] }
    assert.equal((await compactV2Context(initial, { config, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(firstHistory)) + 100, summary })).status, "applied")
    assert.ok(initial.messages.some((message) => message.id === requirement.id))
    const second = { sessionID: "ses_rounds", messages: [...secondHistory] }
    assert.equal((await compactV2Context(second, { config, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(secondHistory)) + 100, summary })).status, "applied")
    assert.equal(calls, 4)
    const catalog = await loadV2Catalog(root, "ses_rounds")
    assert.equal(catalog.entries.length, 2)
    assert.equal(catalog.retirementThrough, 1)
    assert.ok(!second.messages.some((message) => message.id === requirement.id), "first cohort may retire only now")
    assert.ok(second.messages.some((message) => message.id === next.id), "latest user instruction remains literal")
    assert.match(JSON.stringify(second.messages), /Never overwrite user files/)
    const archived = await readV2Archive(root, "ses_rounds", catalog.entries[0]!.id)
    assert.equal(JSON.stringify(archived[0]), JSON.stringify(requirement))
})
