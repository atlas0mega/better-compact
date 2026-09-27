import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import type { Context } from "@opencode/plugin/promise/plugin"
import { appendV2Archive } from "../lib/v2/archive"
import { recallPage, registerV2Recall } from "../lib/v2/recall"

test("bounded V2 pages reconstruct exact UTF-8 content without crossing session/corrupting cursor", () => {
    const original = "🧭 one line\n".repeat(1300)
    const chunks: string[] = []
    let cursor: string | undefined
    do {
        const page = JSON.parse(recallPage("ses_owner", "c000001-aabbccddeeff", original, cursor))
        chunks.push(page.text)
        cursor = page.nextCursor ?? undefined
    } while (cursor)
    assert.equal(chunks.join(""), original)
    const first = JSON.parse(recallPage("ses_owner", "c000001-aabbccddeeff", original))
    assert.throws(() => recallPage("ses_other", "c000001-aabbccddeeff", original, first.nextCursor), /Invalid cursor/)
    assert.throws(() => recallPage("ses_owner", "c000001-aabbccddeeff", original + "changed", first.nextCursor), /Invalid cursor/)
})

test("V2 recall tool only reads the active session's catalog and needs explicit tool permission", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-recall-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const entry = await appendV2Archive(root, "ses_owner", [Message.make({ id: "u1", role: "user",
        content: [Message.text("Sensitive original decision")], })])
    assert.ok(entry)
    let registered: any
    const ctx = {
        tool: { transform: async (fn: (editor: any) => void) => fn({ add: (value: unknown) => { registered = value } }) },
        session: { get: async () => ({ location: { directory: root } }) },
    } as unknown as Context
    await registerV2Recall(ctx)
    assert.equal(registered.options.permission, "better_compact_recall")
    const invoke = (sessionID: string, input: unknown) => registered.execute(input, {
        sessionID, signal: new AbortController().signal,
    })
    const catalog = JSON.parse((await invoke("ses_owner", { mode: "catalog" })).content)
    assert.equal(catalog.entries[0]?.id, entry.id)
    assert.equal(JSON.parse((await invoke("ses_foreign", { mode: "catalog" })).content).entries.length, 0)
    await assert.rejects(invoke("ses_foreign", { mode: "excerpt", archiveId: entry.id, query: "Sensitive" }), /foreign session/)
    assert.match((await invoke("ses_owner", { mode: "excerpt", archiveId: entry.id, query: "Sensitive" })).content, /Sensitive original decision/)
    assert.doesNotMatch((await invoke("ses_owner", { mode: "catalog" })).content, /Sensitive original decision/)
})
