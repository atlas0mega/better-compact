import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp, readFile, writeFile, symlink } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import { appendV2Archive, expireV2Archives, loadV2Catalog, readV2Archive, retiredV2HumanIDs, saveV2Checkpoint, V2_ARCHIVE_RETENTION_MS } from "../lib/v2/archive"
import { identifyV2Messages, restoreV2Messages } from "../lib/v2/identity"

test("V2 virtual identities never alter archived ID-less messages or replay changed tool results", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-idless-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const first = Message.user("Never overwrite user files")
    const repeated = Message.user("Never overwrite user files")
    const result = Message.make({ role: "tool", content: [{ type: "tool-result", id: "call-1", name: "read",
        result: { type: "text", value: "Exact original tool output" } }] })
    const originals = [first, repeated, result]
    const identified = identifyV2Messages(originals)
    assert.equal(new Set(identified.messages.map((message) => message.id)).size, originals.length)
    assert.deepEqual(restoreV2Messages(identified.messages, identified.originals), originals)
    const entry = await appendV2Archive(root, "ses_idless", identified.messages, identified.originals)
    assert.ok(entry)
    assert.equal(JSON.stringify(await readV2Archive(root, "ses_idless", entry.id)), JSON.stringify(originals))
    assert.equal((await readFile(join(root, ".opencode/better-compact/v2/sessions/ses_idless/archives", `${entry.id}.json`), "utf8")),
        JSON.stringify(originals), "archive bytes must contain original messages, not virtual IDs")
    assert.equal(await appendV2Archive(root, "ses_idless", identified.messages, identified.originals), null)
    const corrected = Message.make({ role: "tool", content: [{ type: "tool-result", id: "call-1", name: "read",
        result: { type: "text", value: "Corrected original tool output" } }] })
    const updated = identifyV2Messages([first, repeated, corrected])
    const revision = await appendV2Archive(root, "ses_idless", updated.messages, updated.originals)
    assert.ok(revision)
    assert.equal(JSON.stringify(await readV2Archive(root, "ses_idless", revision.id)), JSON.stringify([corrected]))
    assert.notEqual(entry.messageIDs[2], revision.messageIDs[0])
})

test("V2 archives exact private deltas, idempotently, including opaque provider bytes", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const first = Message.make({ id: "u1", role: "user", content: [Message.text("Never overwrite files")] })
    const state = Message.make({ id: "a1", role: "assistant", content: [
        { type: "reasoning", text: "", encrypted: "opaque-provider-state" },
        { type: "compaction", provider: "openai" as never, encrypted: "opaque-checkpoint" },
    ] })
    const later = Message.make({ id: "u2", role: "user", content: [Message.text("Now fix the tests")] })
    const one = await appendV2Archive(root, "ses_owner", [first, state])
    assert.ok(one)
    assert.equal(await appendV2Archive(root, "ses_owner", [first, state]), null)
    const two = await appendV2Archive(root, "ses_owner", [first, state, later])
    assert.ok(two)
    assert.deepEqual(two.messageIDs, ["u2"])
    assert.equal(JSON.stringify(await readV2Archive(root, "ses_owner", one.id)), JSON.stringify([first, state]))
    assert.equal(JSON.stringify(await readV2Archive(root, "ses_owner", two.id)), JSON.stringify([later]))
    assert.deepEqual((await loadV2Catalog(root, "ses_owner")).entries.map((entry) => entry.id), [one.id, two.id])
    assert.equal((await readFile(join(root, ".opencode/better-compact/.gitignore"), "utf8")), "*\n!.gitignore\n")
    await assert.rejects(readV2Archive(root, "ses_foreign", one.id), /Unknown or expired/)
    await assert.rejects(readV2Archive(root, "../outside", one.id), /Invalid archive session/)
})

test("V2 archive reads reject modified files and symlinked private paths", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const message = Message.make({ id: "user", role: "user", content: [Message.text("private data")] })
    const entry = await appendV2Archive(root, "ses_owner", [message])
    assert.ok(entry)
    const path = join(root, ".opencode", "better-compact", "v2", "sessions", "ses_owner", "archives", `${entry.id}.json`)
    await writeFile(path, "[]")
    await assert.rejects(readV2Archive(root, "ses_owner", entry.id), /Modified archive/)
    await assert.rejects(appendV2Archive(root, "ses_owner", [message]), /Modified archive/)
    const { unlink } = await import("node:fs/promises")
    await unlink(path)
    await symlink(join(root, "other"), path)
    await assert.rejects(readV2Archive(root, "ses_owner", entry.id), /symlink/i)
})

test("concurrent V2 boundary writes serialize without losing either exact delta", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-concurrent-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const left = Message.make({ id: "left", role: "user", content: [Message.text("Keep the left decision")], })
    const right = Message.make({ id: "right", role: "user", content: [Message.text("Keep the right decision")], })
    await Promise.all([appendV2Archive(root, "ses_owner", [left]), appendV2Archive(root, "ses_owner", [right])])
    const catalog = await loadV2Catalog(root, "ses_owner")
    assert.deepEqual(catalog.entries.map((entry) => entry.sequence), [1, 2])
    const recovered = (await Promise.all(catalog.entries.map((entry) => readV2Archive(root, "ses_owner", entry.id)))).flat()
    assert.deepEqual(recovered.map((message) => message.id).sort(), ["left", "right"])
})

test("V2 archive expiry leaves tombstones and never exposes expired raw text", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const message = Message.make({ id: "u1", role: "user", content: [Message.text("old private bytes")] })
    const entry = await appendV2Archive(root, "ses_owner", [message])
    assert.ok(entry)
    const catalogPath = join(root, ".opencode/better-compact/v2/sessions/ses_owner/catalog.json")
    const catalog = await loadV2Catalog(root, "ses_owner")
    catalog.entries[0]!.createdAt = new Date(Date.now() - V2_ARCHIVE_RETENTION_MS - 1000).toISOString()
    await writeFile(catalogPath, JSON.stringify(catalog))
    assert.equal(await expireV2Archives(root, "ses_owner"), 1)
    assert.equal((await loadV2Catalog(root, "ses_owner")).entries[0]?.status, "expired")
    await assert.rejects(readV2Archive(root, "ses_owner", entry.id), /Unknown or expired/)
    assert.equal((await appendV2Archive(root, "ses_owner", [message]))?.sequence, 2)
})

test("V2 expiry finishes a crashed tombstone deletion without trusting a changed file", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const entry = await appendV2Archive(root, "ses_owner", [Message.make({ id: "u1", role: "user",
        content: [Message.text("old private bytes")] })])
    assert.ok(entry)
    const catalog = await loadV2Catalog(root, "ses_owner")
    catalog.entries[0]!.status = "expired"
    const sessionDir = join(root, ".opencode/better-compact/v2/sessions/ses_owner")
    await writeFile(join(sessionDir, "catalog.json"), JSON.stringify(catalog))
    const archivePath = join(sessionDir, "archives", `${entry.id}.json`)
    await writeFile(archivePath, "tampered")
    await assert.rejects(expireV2Archives(root, "ses_owner"), /Modified expired archive/)
    await writeFile(archivePath, JSON.stringify([Message.make({ id: "u1", role: "user",
        content: [Message.text("old private bytes")] })]))
    assert.equal(await expireV2Archives(root, "ses_owner"), 0)
    const { access } = await import("node:fs/promises")
    await assert.rejects(access(archivePath), /ENOENT/)
})

test("V2 catalog only publishes a checkpoint with validated current intent and a real archive", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const entry = await appendV2Archive(root, "ses_owner", [Message.make({ id: "u1", role: "user",
        content: [Message.text("Never overwrite user files")] })])
    assert.ok(entry)
    const base = { rangeHash: "a".repeat(16), archiveID: entry.id, recentUserIntent: ["Never overwrite user files"],
        description: "Feature migration, implementation outcome and useful test lookup clue" }
    await assert.rejects(saveV2Checkpoint(root, "ses_owner", { ...base, handoff: "thin summary" }), /Unvalidated/)
    const handoff = ["## Decisions", "- Preserve existing files.", "## Files & Symbols", "- src/feature.ts",
        "## Errors (verbatim)", "- tests failed", "## What failed and why", "- old implementation dropped context",
        "## Constraints", "- Never overwrite user files", "## Next step", "- Fix tests and verify context."].join("\n")
    await saveV2Checkpoint(root, "ses_owner", { ...base, handoff })
    const catalog = await loadV2Catalog(root, "ses_owner")
    assert.equal(catalog.checkpoint?.rangeHash, base.rangeHash)
    assert.equal(catalog.entries[0]?.status, "ready")
    await assert.rejects(saveV2Checkpoint(root, "ses_owner", { ...base, handoff: handoff + "\nA changed handoff" }), /Conflicting checkpoint/)
    const next = Message.make({ id: "u2", role: "user", content: [Message.text("Finish the tests")] })
    await appendV2Archive(root, "ses_owner", [next])
    assert.deepEqual([...await retiredV2HumanIDs(root, "ses_owner", [Message.make({ id: "u1", role: "user",
        content: [Message.text("Never overwrite user files")] }), next], 1, "Only finish tests")], [])
    assert.deepEqual([...await retiredV2HumanIDs(root, "ses_owner", [Message.make({ id: "u1", role: "user",
        content: [Message.text("Never overwrite user files")] }), next], 1, handoff)], ["u1"])
})
