import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp, readFile, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import type { PluginConfig } from "../lib/v2/config"
import { appendV2Archive, loadV2Catalog, readV2Archive } from "../lib/v2/archive"
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
    assert.match(pointer, /better_compact_recall/)
    assert.doesNotMatch(pointer, /## Compacted Assistant Runs/, "do not duplicate the whole run index in every provider request")
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

test("a stable replay never prunes or rearchives newly appended assistant, tool, or human messages", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-replay-tail-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const older = [Message.user("Keep the exact existing constraints"),
        ...Array.from({ length: 10 }, (_, index) => [
            Message.make({ role: "assistant", content: [{ type: "tool-call", id: `old-${index}`, name: "read", input: { path: `old-${index}` } }] }),
            Message.make({ role: "tool", content: [{ type: "tool-result", id: `old-${index}`, name: "read",
                result: { type: "text", value: `old-${index} ` + "older result ".repeat(220) } }] }),
        ]).flat(), Message.user("Continue the current test")]
    const before = v2Codec.estimateTurns(v2Codec.encode(older))
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const input = { config: local, projectRoot: root, contextLimit: before + 100 }
    const first = { sessionID: "ses_replay_tail", messages: [...older] }
    assert.equal((await compactV2Context(first, input)).status, "applied")
    const firstCatalog = await loadV2Catalog(root, first.sessionID)
    assert.equal(firstCatalog.entries.length, 1)
    assert.ok(firstCatalog.replayFrontier?.prefixSha256)
    const newlyAdded = [
        Message.assistant("New implementation decision after the first replay"),
        Message.make({ role: "assistant", content: [{ type: "tool-call", id: "new-call", name: "read", input: { path: "active.ts" } }] }),
        Message.make({ role: "tool", content: [{ type: "tool-result", id: "new-call", name: "read",
            result: { type: "text", value: "large NEW tool result ".repeat(7_000) } }] }),
        Message.assistant("The latest failure is still being investigated."),
        Message.user("New user correction: do not change the active test."),
    ]
    const appended = { sessionID: first.sessionID, messages: [...older, ...newlyAdded] }
    const result = await compactV2Context(appended, input)
    assert.equal(result.status, "applied")
    assert.deepEqual(appended.messages.slice(-newlyAdded.length), newlyAdded)
    newlyAdded.forEach((message, index) => assert.equal(appended.messages.at(-newlyAdded.length + index), message))
    const replay = { sessionID: first.sessionID, messages: [...older, ...newlyAdded] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(appended.messages), "the same source pays no replay tax")
    assert.equal((await loadV2Catalog(root, first.sessionID)).entries.length, 1, "the appended cohort is not an archive delta")
    const moreWork = Message.assistant("The newest assistant message since replay stays literal too")
    const later = { sessionID: first.sessionID, messages: [...older, ...newlyAdded, moreWork] }
    assert.equal((await compactV2Context(later, input)).status, "applied")
    assert.equal(later.messages.at(-1), moreWork)
    assert.equal((await loadV2Catalog(root, first.sessionID)).entries.length, 1)
})

test("a native checkpoint protects every newer encrypted assistant message even after another human prompt", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-native-tail-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const newer = [Message.user("New task since native replay"),
        Message.make({ role: "assistant", content: [
            { type: "reasoning", text: "", encrypted: "encrypted-since-replay" },
            Message.text("New implementation decision, still exact"),
        ] }),
        Message.user("Newest correction: keep that decision"),
    ]
    const messages = [Message.user("Older constraint"),
        Message.assistant("Older work ".repeat(3_000)),
        Message.assistant([{ type: "compaction", provider: "openai", encrypted: "older-checkpoint" }]),
        ...newer]
    const event = { sessionID: "ses_native_tail", messages: [...messages] }
    const outcome = await compactV2Context(event, { projectRoot: root, config: {
        ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom, prefixSummary: false } },
    }, contextLimit: 15_000, fallbackCeiling: 1_000 })
    assert.equal(outcome.protectedTailMessages, newer.length)
    assert.deepEqual(event.messages.slice(-newer.length), newer)
    if (outcome.status === "applied") {
        assert.equal(JSON.stringify(event.messages.slice(-newer.length)), JSON.stringify(newer))
    }
})

test("an earlier in-place edit cannot replay a historical prefix under the same last ID", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-revised-prefix-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const old = [Message.make({ id: "u1", role: "user", content: [Message.text("Keep old constraint")], }),
        ...Array.from({ length: 12 }, (_, index) => [
            Message.make({ id: `call-${index}`, role: "assistant", content: [{ type: "tool-call", id: `tool-${index}`, name: "read", input: {} }] }),
            Message.make({ id: `result-${index}`, role: "tool", content: [{ type: "tool-result", id: `tool-${index}`, name: "read",
                result: { type: "text", value: `The previous result ${index}. ${"Old log ".repeat(280)}` } }] }),
        ]).flat(),
        Message.make({ id: "u2", role: "user", content: [Message.text("Continue tests")] })]
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const input = { config: local, projectRoot: root, contextLimit: v2Codec.estimateTurns(v2Codec.encode(old)) + 100 }
    const first = { sessionID: "ses_revised", messages: [...old] }
    assert.equal((await compactV2Context(first, input)).status, "applied")
    const edited = [...old]
    edited[2] = Message.make({ id: "result-0", role: "tool", content: [{ type: "tool-result", id: "tool-0", name: "read",
        result: { type: "text", value: "A revised previous result, not the archived one" } }] })
    const replay = { sessionID: first.sessionID, messages: edited }
    const result = await compactV2Context(replay, input)
    assert.equal(result.status, "declined")
    assert.equal(result.reason, "replay-frontier-missing")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(edited))
    assert.equal((await loadV2Catalog(root, first.sessionID)).entries.length, 1)
})

test("older catalogs without a recorded frontier conservatively protect all later work", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-legacy-boundary-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const original = [Message.make({ id: "instruction", role: "user", content: [Message.text("Keep this instruction")] }),
        ...Array.from({ length: 14 }, (_, index) => [
            Message.make({ id: `old-call-${index}`, role: "assistant",
                content: [{ type: "tool-call", id: `old-${index}`, name: "read", input: {} }] }),
            Message.make({ id: `old-result-${index}`, role: "tool",
                content: [{ type: "tool-result", id: `old-${index}`, name: "read",
                    result: { type: "text", value: "Archived or new tool output ".repeat(300) } }] }),
        ]).flat(), Message.make({ id: "latest", role: "user", content: [Message.text("Continue safely")] })]
    const sessionID = "ses_prior_catalog"
    await appendV2Archive(root, sessionID, original.slice(0, 11))
    const file = join(root, ".opencode", "better-compact", "v2", "sessions", sessionID, "catalog.json")
    const catalog = await loadV2Catalog(root, sessionID)
    delete catalog.replayState
    await writeFile(file, JSON.stringify(catalog), { mode: 0o600 })
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const input = { config: local, projectRoot: root, contextLimit: v2Codec.estimateTurns(v2Codec.encode(original)) + 100 }
    const first = { sessionID, messages: [...original] }
    assert.equal((await compactV2Context(first, input)).status, "applied")
    assert.deepEqual(first.messages.slice(-original.length + 11), original.slice(11),
        "all messages newer than the old archive must remain literal")
    const replay = { sessionID, messages: [...original] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(first.messages))
    assert.equal((await loadV2Catalog(root, sessionID)).entries.length, 1)
})

test("a chronological system update stays exact while later tools prune and the replay tail stays raw", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-system-barrier-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const earlier = Message.user("An earlier human request remains literal")
    const authority = Message.make({ role: "system", content: [Message.text("Operator update: do not replace the current instructions")] })
    const old = [earlier, authority, Message.user("Preserve this new user constraint"),
        ...Array.from({ length: 12 }, (_, index) => [
            Message.make({ role: "assistant", content: [{ type: "tool-call", id: `sys-call-${index}`, name: "read", input: { path: `src/${index}` } }] }),
            Message.make({ role: "tool", content: [{ type: "tool-result", id: `sys-call-${index}`, name: "read",
                result: { type: "text", value: `system-era-${index} ` + "historical output ".repeat(500) } }] }),
        ]).flat(), Message.user("Continue the current task")]
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const input = { config: local, projectRoot: root, contextLimit: v2Codec.estimateTurns(v2Codec.encode(old.slice(2))) + 100 }
    const first = { sessionID: "ses_system_barrier", messages: [...old] }
    assert.equal((await compactV2Context(first, input)).status, "applied")
    assert.equal(first.messages[0], earlier)
    assert.equal(first.messages[1], authority)
    assert.equal(first.messages.at(-1), old.at(-1))
    const catalog = await loadV2Catalog(root, first.sessionID)
    assert.equal(catalog.entries.length, 1)
    assert.ok(catalog.entries[0]?.messageIDs.every((id) => !id.startsWith("bcv-0-") && !id.startsWith("bcv-1-")))
    const archived = await readV2Archive(root, first.sessionID, catalog.entries[0]!.id)
    assert.ok(archived.every((message) => message.role !== "system"))
    const newAssistant = Message.assistant("This answer was written after the replay and must stay literal")
    const newCall = Message.make({ role: "assistant", content: [{ type: "tool-call", id: "later-result", name: "read", input: { path: "new.ts" } }] })
    const newTool = Message.make({ role: "tool", content: [{ type: "tool-result", id: "later-result", name: "read",
        result: { type: "text", value: "A new result ".repeat(1_200) } }] })
    const appended = { sessionID: first.sessionID, messages: [...old, newAssistant, newCall, newTool] }
    assert.equal((await compactV2Context(appended, input)).status, "applied")
    assert.deepEqual(appended.messages.slice(-3), [newAssistant, newCall, newTool])
    const replay = { sessionID: first.sessionID, messages: [...old, newAssistant, newCall, newTool] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(appended.messages))
    assert.equal((await loadV2Catalog(root, first.sessionID)).entries.length, 1)
})

test("tool continuations after a system update prune without inventing a new human or losing the old one", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-system-tool-loop-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const human = Message.user("Keep all previous user constraints intact")
    const authority = Message.make({ role: "system", content: [Message.text("Operator update: keep the original task intact")] })
    const older = [human, authority, ...Array.from({ length: 15 }, (_, index) => [
        Message.make({ role: "assistant", content: [{ type: "tool-call", id: `loop-${index}`, name: "read", input: {} }] }),
        Message.make({ role: "tool", content: [{ type: "tool-result", id: `loop-${index}`, name: "read",
            result: { type: "text", value: "Earlier tool observation ".repeat(450) } }] }),
    ]).flat(), Message.assistant("The final implementation decision stays available")]
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: true } } }
    let calls = 0
    const input = { config: local, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(older.slice(2))) + 100,
        summary: { modelContextLimit: 100000, generate: async () => { calls++; return "not needed" } } }
    const event = { sessionID: "ses_system_no_new_user", messages: [...older] }
    const result = await compactV2Context(event, input)
    assert.equal(result.status, "applied")
    assert.equal(result.selectedStrategy, "cheap")
    assert.equal(event.messages[0], human)
    assert.equal(event.messages[1], authority)
    assert.equal(event.messages.at(-1), older.at(-1))
    assert.equal(calls, 0, "do not hallucinate a replacement handoff without a new human after the system barrier")
    const replay = { sessionID: event.sessionID, messages: [...older] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(event.messages))
})

test("a later system update starts a new archive epoch instead of replaying an old checkpoint", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-system-epochs-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const old = [Message.user("Keep the original human instruction"),
        ...Array.from({ length: 10 }, (_, index) => [
            Message.make({ role: "assistant", content: [{ type: "tool-call", id: `epoch-a-${index}`, name: "read", input: {} }] }),
            Message.make({ role: "tool", content: [{ type: "tool-result", id: `epoch-a-${index}`, name: "read",
                result: { type: "text", value: "Old logs ".repeat(350) } }] }),
        ]).flat(), Message.user("Finish the previous tests")]
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const first = { sessionID: "ses_epoch_change", messages: [...old] }
    assert.equal((await compactV2Context(first, { config: local, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(old)) + 100 })).status, "applied")
    const barrier = Message.make({ role: "system", content: [Message.text("New operator instruction supersedes older instructions")], })
    const following = [Message.user("Keep the new user correction"),
        ...Array.from({ length: 10 }, (_, index) => [
            Message.make({ role: "assistant", content: [{ type: "tool-call", id: `epoch-b-${index}`, name: "read", input: {} }] }),
            Message.make({ role: "tool", content: [{ type: "tool-result", id: `epoch-b-${index}`, name: "read",
                result: { type: "text", value: "New logs ".repeat(350) } }] }),
        ]).flat(), Message.user("Continue safely")]
    const full = [...old, barrier, ...following]
    const input = { config: local, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode([...old, ...following])) + 150 }
    const second = { sessionID: first.sessionID, messages: [...full] }
    assert.equal((await compactV2Context(second, input)).status, "applied")
    full.slice(0, old.length + 1).forEach((message, index) => assert.equal(second.messages[index], message))
    assert.equal(second.messages.at(-1), full.at(-1))
    const catalog = await loadV2Catalog(root, first.sessionID)
    assert.equal(catalog.entries.length, 2)
    assert.notEqual(catalog.entries[0]?.authorityKey, catalog.entries[1]?.authorityKey)
    assert.equal(catalog.replayFrontier?.authorityKey, catalog.entries[1]?.authorityKey)
    const replay = { sessionID: first.sessionID, messages: [...full] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(second.messages))
    assert.equal((await loadV2Catalog(root, first.sessionID)).entries.length, 2)
})

test("a replacement handoff after a system update cannot reuse the prior authority checkpoint", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-handoff-authority-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const previous = [Message.user("Retain the original task constraint"),
        ...Array.from({ length: 12 }, (_, index) => Message.assistant(
            `Earlier decision ${index}; ${"prior diagnostic ".repeat(180)}`)),
        Message.user("Complete the first test group")]
    const authority = Message.make({ role: "system", content: [Message.text("New authority: verify the second test group")] })
    const subsequent = [Message.user("Retain this new constraint and verify the second test group"),
        ...Array.from({ length: 12 }, (_, index) => Message.assistant(
            `Second test decision ${index}; ${"new diagnostic ".repeat(180)}`)),
        Message.user("Finish the second test group")]
    const handoff = (which: string) => ["## Decisions", `- Test group ${which} remains in progress.`,
        "## Files & Symbols", "- src/feature.ts and tests/feature.test.ts.",
        "## Errors (verbatim)", "- No confirmed regression.",
        "## What failed and why", "- Routine diagnostics hid the current test decision.",
        "## Constraints", which === "first" ? "- Retain the original task constraint." :
            "- Retain this new constraint and verify the second test group.",
        "## Next step", which === "first" ? "- Complete the first test group." : "- Finish the second test group."].join("\n")
    let calls = 0
    const summary = { modelContextLimit: 100000, generate: async (_prompt: string, variant?: string) => {
        calls++
        return variant ? JSON.stringify({ handoff: handoff(calls <= 2 ? "first" : "second") }) :
            "Implementation diagnostics and the required test group progression"
    } }
    const first = { sessionID: "ses_authority_handoff", messages: [...previous] }
    assert.equal((await compactV2Context(first, { config, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(previous)) + 100, summary })).status, "applied")
    const originalCheckpoint = (await loadV2Catalog(root, first.sessionID)).checkpoint
    assert.ok(originalCheckpoint)
    const full = [...previous, authority, ...subsequent]
    const input = { config, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode([...previous, ...subsequent])) + 200, summary }
    const second = { sessionID: first.sessionID, messages: [...full] }
    const result = await compactV2Context(second, input)
    assert.equal(result.status, "applied")
    assert.equal(result.selectedStrategy, "new-handoff")
    assert.equal(calls, 4)
    full.slice(0, previous.length + 1).forEach((message, index) => assert.equal(second.messages[index], message))
    assert.equal(second.messages.at(-1), subsequent.at(-1))
    const catalog = await loadV2Catalog(root, first.sessionID)
    assert.equal(catalog.entries.length, 2)
    assert.notEqual(catalog.checkpoint?.authorityKey, originalCheckpoint.authorityKey)
    assert.equal(catalog.checkpoint?.authorityKey, catalog.replayFrontier?.authorityKey)
    const replay = { sessionID: first.sessionID, messages: [...full] }
    assert.equal((await compactV2Context(replay, input)).status, "applied")
    assert.equal(calls, 4)
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(second.messages))
})

test("V2 seam leaves below-trigger requests untouched", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-context-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const message = Message.make({ id: "active", role: "user", content: [Message.text("Continue carefully")] })
    const event = { sessionID: "ses_small", messages: [message] }
    assert.equal((await compactV2Context(event, { config, projectRoot: root, contextLimit: 200000 })).status, "below-trigger")
    assert.deepEqual(event.messages, [message])
})

test("a host fallback ceiling never publishes an oversized pruned request or replay frontier", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-native-fallback-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const source = [Message.user("Never overwrite user files"),
        ...Array.from({ length: 12 }, (_, index) => [
            Message.make({ role: "assistant", content: [{ type: "tool-call", id: `fallback-${index}`, name: "read", input: {} }] }),
            Message.make({ role: "tool", content: [{ type: "tool-result", id: `fallback-${index}`, name: "read",
                result: { type: "text", value: "Old diagnostics ".repeat(300) } }] }),
        ]).flat(), Message.user("Current task and constraints ".repeat(1_000))]
    const local = { ...config, compaction: { ...config.compaction, custom: { ...config.compaction.custom,
        recentToolTokens: 0, prefixSummary: false } } }
    const input = { config: local, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(source)) + 100, fallbackCeiling: 500 }
    const first = { sessionID: "ses_native_fallback", messages: [...source] }
    const result = await compactV2Context(first, input)
    assert.equal(result.reason, "needs-native-fallback")
    assert.equal(result.status, "declined")
    assert.equal(JSON.stringify(first.messages), JSON.stringify(source))
    const catalog = await loadV2Catalog(root, first.sessionID)
    assert.equal(catalog.entries.length, 1)
    assert.equal(catalog.replayState, "unapplied")
    assert.equal(catalog.replayFrontier, undefined)
    assert.equal(JSON.stringify(await readV2Archive(root, first.sessionID, catalog.entries[0]!.id)),
        JSON.stringify(source.slice(0, catalog.entries[0]!.messageIDs.length)))
    const replay = { sessionID: first.sessionID, messages: [...source] }
    assert.equal((await compactV2Context(replay, input)).reason, "needs-native-fallback")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(source))
    assert.equal((await loadV2Catalog(root, first.sessionID)).entries.length, 1)
})

test("a validated Luna draft is not published when only native fallback can fit", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-handoff-fallback-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const messages = [Message.user("Never overwrite user files"),
        ...Array.from({ length: 12 }, (_, index) => Message.assistant(
            `Implementation ${index}: ${"Old diagnostic detail ".repeat(200)}`)),
        Message.user("Finish the active tests")]
    const handoff = ["## Decisions", "- Preserve existing files and complete the implementation.",
        "## Files & Symbols", "- src/feature.ts and tests/feature.test.ts remain active.",
        "## Errors (verbatim)", "- Old diagnostic detail was resolved.",
        "## What failed and why", "- Repeated historical output hid the next action.",
        "## Constraints", "- Never overwrite user files.", "## Next step", "- Finish the active tests."].join("\n")
    let calls = 0
    const input = { config, projectRoot: root, contextLimit: v2Codec.estimateTurns(v2Codec.encode(messages)) + 100,
        fallbackCeiling: 100, summary: { modelContextLimit: 100000,
            generate: async (_prompt: string, variant: string | undefined) => {
                calls++
                return variant ? JSON.stringify({ handoff }) : "Validated implementation, historical diagnostics and tests"
            } } }
    const event = { sessionID: "ses_handoff_fallback", messages: [...messages] }
    assert.equal((await compactV2Context(event, input)).reason, "needs-native-fallback")
    assert.equal(JSON.stringify(event.messages), JSON.stringify(messages))
    const catalog = await loadV2Catalog(root, event.sessionID)
    assert.equal(catalog.checkpoint, undefined, "a handoff the model never saw cannot become the replay checkpoint")
    assert.equal(catalog.replayFrontier, undefined)
    assert.equal(catalog.summaryAttempt?.reason, "needs_native_fallback")
    assert.equal(calls, 2)
    assert.equal((await compactV2Context({ sessionID: event.sessionID, messages: [...messages] }, input)).reason,
        "needs-native-fallback")
    assert.equal(calls, 2, "unchanged fallback must not bill the handoff again")
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
    const secondOutcome = await compactV2Context(second, { config, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(secondHistory)) + 100, summary })
    assert.equal(secondOutcome.status, "applied", JSON.stringify({ secondOutcome, calls,
        catalog: await loadV2Catalog(root, "ses_rounds") }))
    assert.equal(calls, 4)
    const catalog = await loadV2Catalog(root, "ses_rounds")
    assert.equal(catalog.entries.length, 2)
    assert.equal(catalog.retirementThrough, 1)
    assert.ok(!second.messages.some((message) => message.id === requirement.id), "first cohort may retire only now")
    assert.ok(second.messages.some((message) => message.id === next.id), "latest user instruction remains literal")
    assert.match(JSON.stringify(second.messages), /Never overwrite user files/)
    const archived = await readV2Archive(root, "ses_rounds", catalog.entries[0]!.id)
    assert.equal(JSON.stringify(archived[0]), JSON.stringify(requirement))
    const replay = { sessionID: second.sessionID, messages: [...secondHistory] }
    assert.equal((await compactV2Context(replay, { config, projectRoot: root,
        contextLimit: v2Codec.estimateTurns(v2Codec.encode(secondHistory)) + 100, summary })).status, "applied")
    assert.equal(calls, 4, "the second handoff must replay without another Luna call")
    assert.equal(JSON.stringify(replay.messages), JSON.stringify(second.messages), "round two must also replay byte-identically")
})
