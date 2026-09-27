import assert from "node:assert/strict"
import test from "node:test"
import { Message } from "@opencode/ai"
import { buildPlan, rangeHash, reasoningStage, transformTurns } from "@better-compact/core"
import { v2Codec, v2Conventions, v2Spec } from "../lib/v2/codec"
import { validateRequestTransform } from "../lib/v2/safety"

test("V2 codec keeps opaque provider checkpoints and encrypted reasoning byte-exact", () => {
    const user = Message.make({ id: "msg_user", role: "user", content: [{ type: "text", text: "Keep the user decision" }] })
    const assistant = Message.make({ id: "msg_assistant", role: "assistant", content: [
        { type: "reasoning", text: "", encrypted: "opaque-reasoning-byte-sequence" },
        { type: "compaction", provider: "openai" as never, encrypted: "opaque-provider-checkpoint" },
        { type: "text", text: "Work completed" },
    ] })
    const turns = v2Codec.encode([user, assistant])
    assert.equal(turns.length, 2)
    assert.equal(turns[0]?.role, "user")
    assert.equal(turns[1]?.items[0]?.kind, "reasoning")
    assert.equal(v2Conventions.isPreservedItem?.(turns[1]!.items[0]!), true)
    assert.equal(v2Conventions.isPreservedItem?.(turns[1]!.items[1]!), true)
    assert.deepEqual(v2Codec.decode(turns, [user, assistant])[1]?.content, assistant.content)
    assert.match(v2Codec.transcriptDocument!(turns), /opaque-provider-checkpoint/)
})

test("reasoning pruning never discards encrypted provider state needed for continuation", () => {
    const message = Message.assistant([
        { type: "reasoning", text: "ordinary reasoning" },
        { type: "reasoning", text: "", encrypted: "opaque-provider-state" },
        { type: "text", text: "Keep working" },
    ])
    const turns = v2Codec.encode([message])
    const changed = reasoningStage.run(turns, {
        rawTailStartIndex: 1,
        preservedReasoningItemKeys: new Set(),
        conventions: v2Conventions,
    } as never)
    assert.equal(changed.changedItems, 1)
    assert.equal(turns[0]?.items.length, 2)
    assert.equal((turns[0]?.items[0]?.handle as { encrypted?: string }).encrypted, "opaque-provider-state")
})

test("V2 codec retains linked tool call/result identity and keeps the latest human turn", () => {
    const call = Message.make({ id: "msg_call", role: "assistant", content: [{ type: "tool-call", id: "call_1", name: "read", input: { path: "src/a.ts" } }] })
    const result = Message.make({ id: "msg_result", role: "tool", content: [{ type: "tool-result", id: "call_1", name: "read", result: { type: "text", value: "content" } }] })
    const user = Message.make({ id: "msg_user", role: "user", content: [{ type: "text", text: "Next fix the tests" }] })
    const turns = v2Codec.encode([call, result, user])
    assert.equal(turns[0]?.items[0]?.kind, "tool")
    assert.equal(turns[1]?.items[0]?.kind, "tool")
    assert.equal((turns[0]?.items[0] as { callId: string }).callId, "call_1")
    assert.equal((turns[1]?.items[0] as { callId: string }).callId, "call_1")
    assert.equal(turns[2]?.role, "user")
    assert.deepEqual(v2Codec.decode(turns, [call, result, user]).map((message) => message.role), ["assistant", "tool", "user"])
})

test("V2 codec fails closed rather than dropping operator-authored system updates", () => {
    const instruction = Message.system("Do not lose this decision")
    assert.throws(() => v2Codec.encode([instruction, Message.user("Continue")]), /must bypass Better Compact pruning/)
})

test("V2 range identity changes for revised tool output but not accounting metadata", () => {
    const original = Message.make({ id: "same", role: "tool", content: [
        { type: "tool-result", id: "call", name: "read", result: { type: "text", value: "old" } },
    ], metadata: { tokens: 1 } })
    const revised = Message.make({ ...original, content: [
        { type: "tool-result", id: "call", name: "read", result: { type: "text", value: "new" } },
    ] })
    assert.notEqual(rangeHash(v2Codec.encode([original])), rangeHash(v2Codec.encode([revised])))
    assert.equal(rangeHash(v2Codec.encode([original])), rangeHash(v2Codec.encode([Message.make({
        ...original, metadata: { tokens: 100 },
    })])))
})

test("a V2 ladder plan reduces old tool payloads without losing the current user task", () => {
    const old = Array.from({ length: 8 }, (_, index) => [
        Message.make({ id: `call-${index}`, role: "assistant", content: [{ type: "tool-call", id: `id-${index}`, name: "read", input: { file: `src/${index}.ts` } }] }),
        Message.make({ id: `result-${index}`, role: "tool", content: [{ type: "tool-result", id: `id-${index}`, name: "read", result: { type: "text", value: "old file content ".repeat(180) } }] }),
    ]).flat()
    const messages = [...old, Message.make({ id: "current", role: "user", content: [{ type: "text", text: "Fix the active failing test, not the old files" }] })]
    const turns = v2Codec.encode(messages)
    const before = v2Codec.estimateTurns(turns)
    const plan = buildPlan(turns, {
        sessionKey: "ses_fixture", citablePath: () => ".opencode/better-compact/private/fixture.json",
        contextLimit: Math.max(1000, before + 100), triggerTokens: 1, targetTokens: Math.floor(before * .55),
        minTailMessages: 1, minTailUserTurns: 1, force: true, summariesAllowed: false, prefixSummaryAllowed: false,
    }, v2Spec)
    assert.ok(plan, "old tool payloads should be eligible for a plan")
    const transformed = v2Codec.decode(transformTurns(turns, 0, plan, v2Spec), messages)
    assert.equal(transformed.at(-1)?.role, "user")
    assert.equal(transformed.at(-1)?.content[0]?.type, "text")
    assert.match((transformed.at(-1)?.content[0] as { text: string }).text, /Fix the active failing test/)
    assert.ok(v2Codec.estimateTurns(v2Codec.encode(transformed)) < before, "provider-visible context must shrink")
    const calls = transformed.flatMap((message) => message.content.filter((part) => part.type === "tool-call").map((part) => part.id))
    const results = transformed.flatMap((message) => message.content.filter((part) => part.type === "tool-result").map((part) => part.id))
    assert.deepEqual(calls.sort(), results.sort(), "tool calls and results must remain paired")
    validateRequestTransform(messages, transformed)
})

test("V2 compaction keeps older human constraints when no validated handoff exists", () => {
    const messages = [
        Message.make({ id: "requirement", role: "user", content: [{ type: "text", text: "Never overwrite user files; keep output isolated." }] }),
        ...Array.from({ length: 8 }, (_, index) => Message.make({ id: `old-${index}`, role: "tool", content: [
            { type: "tool-result", id: `call-${index}`, name: "read", result: { type: "text", value: "irrelevant old log ".repeat(160) } },
        ] })),
        Message.make({ id: "correction", role: "user", content: [{ type: "text", text: "Correction: keep the agent's latest test decision too." }] }),
        Message.make({ id: "current", role: "user", content: [{ type: "text", text: "Continue implementing the requested feature." }] }),
    ]
    const turns = v2Codec.encode(messages)
    const before = v2Codec.estimateTurns(turns)
    const plan = buildPlan(turns, {
        sessionKey: "ses_constraints", citablePath: () => ".opencode/better-compact/private/constraints.json",
        contextLimit: before + 100, triggerTokens: 1, targetTokens: Math.floor(before * .55), force: true,
        minTailMessages: 1, minTailUserTurns: 1, summariesAllowed: false, prefixSummaryAllowed: false,
    }, v2Spec)
    assert.ok(plan)
    const output = v2Codec.decode(transformTurns(turns, 0, plan, v2Spec), messages)
    const text = output.flatMap((message) => message.content.filter((part) => part.type === "text").map((part) => part.text)).join("\n")
    assert.match(text, /Never overwrite user files/)
    assert.match(text, /keep the agent's latest test decision/)
    assert.match(text, /Continue implementing the requested feature/)
})
