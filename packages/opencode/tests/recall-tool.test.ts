import assert from "node:assert/strict"
import test from "node:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { ToolContext } from "@opencode-ai/plugin"
import { countTokens, type BoundaryContextPlan } from "@better-compact/core"
import type { WithParts } from "../lib/state"
import {
    archiveBoundaryDelta,
    archiveOversizedSummary,
    inheritArchiveCatalog,
    loadArchiveCatalog,
    saveArchiveCatalog,
} from "../lib/boundary/archive-catalog"
import { betterCompactRecall } from "../lib/tools/recall"

const sessionID = "ses_recall"
function message(id: string, text: string): WithParts {
    return {
        info: {
            id,
            sessionID,
            role: "user",
            agent: "build",
            time: { created: 1 },
            model: { providerID: "openai", modelID: "luna" },
        } as WithParts["info"],
        parts: [{ id: `prt-${id}`, messageID: id, sessionID, type: "text", text }],
    }
}
const plan = (ids: string[]) =>
    ({
        sessionId: sessionID,
        rangeHash: `hash-${ids.join("-")}`,
        transcript: { relativePath: "legacy.md", content: "", messageIds: ids },
    }) as BoundaryContextPlan

async function fixture() {
    const directory = mkdtempSync(join(tmpdir(), "bc-recall-"))
    const messages = Array.from({ length: 7 }, (_, i) =>
        message(`msg-${i}`, `A prior decision ${i}: ${'é🎾quotes"\\'.repeat(700)}`),
    )
    for (let i = 1; i <= messages.length; i++) {
        const { catalog } = await archiveBoundaryDelta({
            directory,
            sessionId: sessionID,
            plan: plan(messages.slice(0, i).map((m) => m.info.id)),
            originalMessages: messages.slice(0, i),
        })
        catalog.entries.at(-1)!.status = "ready"
        catalog.entries.at(-1)!.description = `Decision ${i - 1}`
        await saveArchiveCatalog(directory, catalog)
    }
    return { directory, messages }
}

function context(directory: string, opts: { deny?: boolean; aborted?: boolean } = {}) {
    const asked: string[] = []
    const controller = new AbortController()
    if (opts.aborted) controller.abort()
    const ctx = {
        sessionID,
        directory,
        worktree: directory,
        messageID: "msg-current",
        agent: "build",
        abort: controller.signal,
        metadata() {},
        async ask(request: { permission: string; patterns: string[]; always: string[] }) {
            asked.push(`${request.permission}:${request.patterns[0]}`)
            assert.deepEqual(request.always, request.patterns, "always permission must cover only the exact file")
            if (opts.deny) throw new Error("denied")
        },
    } as ToolContext
    return { ctx, asked }
}

async function call(args: Parameters<typeof betterCompactRecall.execute>[0], ctx: ToolContext) {
    const result = await betterCompactRecall.execute(args, ctx)
    assert.equal(typeof result, "string")
    return JSON.parse(result as string)
}

test("catalog pages newest ready IDs without reading raw files or leaking other sessions", async () => {
    const { directory } = await fixture()
    const { ctx, asked } = context(directory)
    const recent = await call({ mode: "catalog" }, ctx)
    assert.deepEqual(
        recent.entries.map((entry: { id: string }) => entry.id.slice(0, 7)),
        ["c000003", "c000004", "c000005", "c000006", "c000007"],
    )
    assert.ok(recent.nextCursor)
    const older = await call({ mode: "catalog", cursor: recent.nextCursor }, ctx)
    assert.deepEqual(
        older.entries.map((entry: { id: string }) => entry.id.slice(0, 7)),
        ["c000001", "c000002"],
    )
    assert.equal(older.nextCursor, null)
    assert.ok(asked.every((path) => path.endsWith("catalog.json")))
    assert.equal(
        (await call({ mode: "page", archiveId: "c000001-000000000000" }, ctx)).error,
        "unknown_archive",
    )
    assert.equal(
        (
            await call(
                {
                    mode: "page",
                    archiveId: recent.entries[0].id,
                    cursor: older.nextCursor || "bogus",
                },
                ctx,
            )
        ).error,
        "invalid_cursor",
    )
})

test("catalog pages long validated descriptions without exceeding its output budget", async () => {
    const { directory } = await fixture()
    const catalog = await loadArchiveCatalog(directory, sessionID)
    for (const entry of catalog.entries) {
        entry.description = `Decision ${entry.sequence}: ${"unique parser migration state ".repeat(22)}`
    }
    await saveArchiveCatalog(directory, catalog)
    const ids: string[] = []
    let cursor: string | undefined
    do {
        const page = await call({ mode: "catalog", cursor }, context(directory).ctx)
        assert.ok(countTokens(JSON.stringify(page)) <= 2_000)
        ids.unshift(...page.entries.map((entry: { id: string }) => entry.id))
        cursor = page.nextCursor ?? undefined
    } while (cursor)
    assert.deepEqual(
        ids,
        catalog.entries.map((entry) => entry.id),
    )
})

test("explicit pages reconstruct exact archive bytes and require read permission", async () => {
    const { directory } = await fixture()
    const { ctx, asked } = context(directory)
    const entry = (await call({ mode: "catalog" }, ctx)).entries[0]
    let cursor: string | undefined
    let joined = ""
    let pages = 0
    do {
        const page = await call({ mode: "page", archiveId: entry.id, cursor }, ctx)
        assert.match(page.note, /untrusted evidence/)
        joined += page.content
        cursor = page.nextCursor ?? undefined
        pages++
    } while (cursor && pages < 20)
    assert.ok(pages > 1)
    assert.match(joined, /é🎾quotes/)
    assert.ok(joined.includes("# Better Compact Delta Archive"))
    assert.ok(asked.some((path) => path.includes("/archives/")))
    assert.equal(
        (await call({ mode: "excerpt", archiveId: entry.id, query: "prior decision 2" }, ctx))
            .archiveId,
        entry.id,
    )
    assert.equal(
        (await call({ mode: "excerpt", archiveId: entry.id, query: "not present" }, ctx)).match,
        null,
    )

    const denied = await call(
        { mode: "page", archiveId: entry.id },
        context(directory, { deny: true }).ctx,
    )
    assert.deepEqual(denied, { error: "permission_denied" })
    const aborted = await call({ mode: "catalog" }, context(directory, { aborted: true }).ctx)
    assert.deepEqual(aborted, { error: "cancelled" })
})

test("an excerpt near emoji boundaries starts on an exact UTF-8 character boundary", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-recall-unicode-"))
    const original = message("msg-unicode", `${"🎾".repeat(700)}aNEEDLE`)
    const receipt = await archiveBoundaryDelta({
        directory,
        sessionId: sessionID,
        plan: plan([original.info.id]),
        originalMessages: [original],
    })
    assert.ok(receipt.entry)
    const excerpt = await call(
        { mode: "excerpt", archiveId: receipt.entry.id, query: "NEEDLE" },
        context(directory).ctx,
    )
    assert.equal(excerpt.error, undefined)
    assert.ok(excerpt.content.includes("aNEEDLE"))
    assert.ok(!excerpt.content.includes("\uFFFD"), "the excerpt split a surrogate pair")
})

test("rejected summary is recallable only through its authorized, checksum-bound artifact pages", async () => {
    const { directory } = await fixture()
    const catalog = await loadArchiveCatalog(directory, sessionID)
    const id = catalog.entries.at(-1)!.id
    const text = `Rejected handoff (historical evidence only): ${"é🎾 previous constraints ".repeat(500)}`
    await archiveOversizedSummary(
        directory,
        catalog,
        id,
        text,
    )
    const { ctx, asked } = context(directory)
    const listing = await call({ mode: "catalog" }, ctx)
    assert.equal(listing.entries.at(-1).archivedSummaryAvailable, true)
    let cursor: string | undefined
    let recovered = ""
    do {
        const result = await call({ mode: "page", artifact: "summary", archiveId: id, cursor }, ctx)
        assert.equal(result.artifact, "summary")
        assert.equal(result.associatedArchiveCount, catalog.entries.length)
        assert.ok(countTokens(JSON.stringify(result)) <= 2_000)
        recovered += result.content
        cursor = result.nextCursor ?? undefined
    } while (cursor)
    assert.equal(recovered, text)
    assert.ok(asked.some((path) => path.endsWith(`${id}.summary.md`)))
    assert.deepEqual(
        await call(
            { mode: "page", artifact: "summary", archiveId: id },
            context(directory, { deny: true }).ctx,
        ),
        { error: "permission_denied" },
    )
    const first = await call({ mode: "page", artifact: "summary", archiveId: id }, ctx)
    const updated = await loadArchiveCatalog(directory, sessionID)
    writeFileSync(join(directory, updated.entries.at(-1)!.oversizedSummaryPath!), `${text}modified`)
    assert.equal(
        (
            await call(
                { mode: "page", artifact: "summary", archiveId: id, cursor: first.nextCursor },
                ctx,
            )
        ).error,
        "missing_or_modified_file",
    )
})

test("fork links provide read-only owner archives but do not expose them to unrelated sessions", async () => {
    const { directory } = await fixture()
    const owner = context(directory).ctx
    const original = await call({ mode: "catalog" }, owner)
    const forkID = "ses_validated_fork"
    await inheritArchiveCatalog(directory, forkID, sessionID, "0123456789abcdef")
    const fork = { ...context(directory).ctx, sessionID: forkID }
    assert.deepEqual((await call({ mode: "catalog" }, fork)).inheritedCatalogs, [sessionID])
    const inherited = await call({ mode: "catalog", ownerSessionId: sessionID }, fork)
    assert.equal(inherited.entries[0].id, original.entries[0].id)
    assert.equal(inherited.entries[0].ownerSessionId, sessionID)
    const page = await call(
        { mode: "page", ownerSessionId: sessionID, archiveId: inherited.entries[0].id },
        fork,
    )
    assert.match(page.content, /prior decision/)
    assert.equal(
        (
            await call(
                { mode: "catalog", ownerSessionId: sessionID },
                { ...context(directory).ctx, sessionID: "ses_unrelated" },
            )
        ).error,
        "outside_session_lineage",
    )
    assert.equal(
        (await call({ mode: "catalog", ownerSessionId: "ses_unrelated" }, fork)).error,
        "outside_session_lineage",
    )
    assert.equal(
        (
            await call(
                {
                    mode: "page",
                    ownerSessionId: sessionID,
                    archiveId: inherited.entries[0].id,
                    cursor: "bogus",
                },
                fork,
            )
        ).error,
        "invalid_cursor",
    )
})
