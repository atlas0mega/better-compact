import assert from "node:assert/strict"
import test from "node:test"
import { Message } from "@opencode/ai"
import { summarizeV2Boundary, v2Evidence, validateV2Handoff } from "../lib/v2/handoff"

const handoff = ["## Decisions", "- Keep user decisions and preserve reasons.", "## Files & Symbols",
    "- src/feature.ts is the active implementation.", "## Errors (verbatim)", "- Tests initially failed.",
    "## What failed and why", "- The first attempt lost tool context.", "## Constraints",
    "- Never overwrite user files; preserve exact wording and tool provenance.", "## Next step",
    "- Finish the active tests and verify the provider-visible request."].join("\n")

test("V2 handoff validator rejects a thin or constraint-losing output", () => {
    assert.equal(validateV2Handoff(JSON.stringify({ handoff: "short" }), ["Never overwrite user files"]), null)
    assert.equal(validateV2Handoff(JSON.stringify({ handoff: handoff.replace("Never overwrite user files", "Do something else") }),
        ["Never overwrite user files"]), null)
    assert.equal(validateV2Handoff(JSON.stringify({ handoff }), ["Never overwrite user files"]), handoff)
})

test("V2 evidence keeps entire human wording but never sends encrypted provider bytes", () => {
    const source = [Message.make({ id: "u1", role: "user", content: [Message.text("Keep the exact older user wording")] }),
        Message.make({ id: "a1", role: "assistant", content: [
            { type: "reasoning", text: "", encrypted: "private-provider-state" },
            { type: "tool-result", id: "t1", name: "read", result: { type: "text", value: "large log ".repeat(200) } },
        ] })]
    const evidence = v2Evidence(source).join("\n")
    assert.match(evidence, /Keep the exact older user wording/)
    assert.doesNotMatch(evidence, /private-provider-state/)
    assert.ok(evidence.length < 1500)
})

test("one fitting Luna/high handoff and default-effort description use two calls", async () => {
    const calls: Array<"high" | undefined> = []
    const result = await summarizeV2Boundary({
        messages: [Message.make({ id: "u1", role: "user", content: [Message.text("Never overwrite user files")] })],
        archiveID: "c000001-aabbccddeeff", recentUserIntent: ["Never overwrite user files"], modelContextLimit: 100000,
        generate: async (_prompt, variant) => {
            calls.push(variant)
            return variant === "high" ? JSON.stringify({ handoff }) : "Feature migration, failed tests, and the decision not to overwrite user files"
        },
    })
    assert.equal(result.ok, true)
    assert.deepEqual(calls, ["high", undefined])
    assert.equal(result.calls, 2)
})

test("oversized human evidence is never clipped into a fake successful summary", async () => {
    let calls = 0
    const result = await summarizeV2Boundary({
        messages: [Message.make({ id: "u1", role: "user", content: [Message.text("Unique critical wording ".repeat(15000))] })],
        archiveID: "c000001-aabbccddeeff", recentUserIntent: [], modelContextLimit: 27000,
        generate: async () => { calls++; return JSON.stringify({ handoff }) },
    })
    assert.equal(result.ok, false)
    assert.equal(calls, 0)
})

test("V2 large evidence fans out into bounded concurrent source calls then validates one grand handoff", async () => {
    const messages = Array.from({ length: 35 }, (_, index) => Message.make({ id: `evidence-${index}`,
        role: "assistant", content: [Message.text(`Decision ${index}: ${"significant chronological progress ".repeat(65)}`)] }))
    const sourcePrompts: string[] = []
    const variants: Array<string | undefined> = []
    const result = await summarizeV2Boundary({
        messages, archiveID: "c000001-aabbccddeeff", recentUserIntent: [], modelContextLimit: 29000,
        generate: async (prompt, variant) => {
            variants.push(variant)
            if (prompt.includes("Summarize only this chunk")) {
                sourcePrompts.push(prompt)
                return JSON.stringify({ handoff })
            }
            return variant === undefined ? "Feature decisions, tests and the implementation trajectory" : JSON.stringify({ handoff })
        },
    })
    assert.equal(result.ok, true)
    assert.ok(sourcePrompts.length >= 2 && sourcePrompts.length <= 5)
    assert.equal(result.calls, sourcePrompts.length + 2)
    assert.ok(result.calls <= 7)
    for (const message of messages) assert.ok(sourcePrompts.some((prompt) => prompt.includes(`"${message.id}"`)),
        `missing ${message.id} from bounded source evidence`)
    assert.equal(variants.at(-1), undefined, "description must use the default variant")
})
