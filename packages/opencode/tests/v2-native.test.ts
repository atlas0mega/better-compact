import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import { loadV2Catalog, readV2Archive } from "../lib/v2/archive"
import { compactV2Native } from "../lib/v2/native"

test("manual V2 compaction produces a validated checkpoint and exact archived originals", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-native-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const messages = [Message.make({ id: "u1", role: "user", content: [Message.text("Never overwrite user files")] }),
        ...Array.from({ length: 6 }, (_, index) => Message.make({ id: `a${index}`, role: "assistant",
            content: [Message.text(`Implemented stage ${index} with detailed reasoning ${"step ".repeat(200)}`)] })),
        Message.make({ id: "u2", role: "user", content: [Message.text("Finish the active tests")] })]
    const handoff = ["## Decisions", "- Implementation stages now exist.", "## Files & Symbols", "- src/feature.ts",
        "## Errors (verbatim)", "- tests failed earlier", "## What failed and why", "- repeated progress hid a failure",
        "## Constraints", "- Never overwrite user files", "## Next step", "- Finish the active tests"].join("\n")
    const output = await compactV2Native({ root, sessionID: "ses_manual", messages, summaryModelLimit: 100000,
        generate: async (_prompt, variant) => variant === "high" ? JSON.stringify({ handoff })
            : "Implementation stages, earlier test failure and next test action" })
    assert.equal(output, handoff)
    const catalog = await loadV2Catalog(root, "ses_manual")
    assert.equal(catalog.entries[0]?.status, "ready")
    assert.equal(catalog.checkpoint?.handoff, handoff)
    assert.equal(JSON.stringify(await readV2Archive(root, "ses_manual", catalog.entries[0]!.id)), JSON.stringify(messages))
})

test("manual V2 compaction rejects a missing older user requirement without overwriting history", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-native-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const messages = [Message.make({ id: "u1", role: "user", content: [Message.text("Never overwrite user files")] }),
        Message.make({ id: "a1", role: "assistant", content: [Message.text("Progress ".repeat(600))] })]
    const source = JSON.stringify(messages)
    await assert.rejects(compactV2Native({ root, sessionID: "ses_manual", messages, summaryModelLimit: 100000,
        generate: async () => JSON.stringify({ handoff: "Done" }) }), /V2 handoff declined: invalid_output/)
    assert.equal(JSON.stringify(messages), source)
    assert.equal((await loadV2Catalog(root, "ses_manual")).entries[0]?.status, "pending")
})
