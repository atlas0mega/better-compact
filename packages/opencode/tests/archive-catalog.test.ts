import assert from "node:assert/strict"
import test from "node:test"
import {
    existsSync,
    mkdirSync,
    mkdtempSync,
    readFileSync,
    rmSync,
    symlinkSync,
    writeFileSync,
} from "node:fs"
import { tmpdir } from "node:os"
import { dirname, join } from "node:path"
import { createEngine, type BoundaryContextPlan, type PlanSnapshot } from "@better-compact/core"
import type { WithParts } from "../lib/state"
import {
    archiveBoundaryDelta,
    archiveCatalogPath,
    archiveOversizedSummary,
    ARCHIVE_RETENTION_MS,
    expireArchives,
    loadArchiveCatalog,
    readArchiveEntry,
    readArchivedSummary,
    projectArchiveRoots,
    saveArchiveCatalog,
    uncoveredMessages,
} from "../lib/boundary/archive-catalog"
import { readPrivateFile } from "../lib/private-storage"
import { openCodeCodec, openCodeSpec } from "../lib/codec"
import { retainArchivedUserText } from "../lib/boundary/engine"
import { startArchiveDescriptionBackfill } from "../lib/hooks"
import { Logger } from "../lib/logger"
import { createRuntimeState } from "../lib/state"

const sessionId = "ses_archive_test"

function native(id: string, role: "user" | "assistant", text: string, created: number): WithParts {
    return {
        info: {
            id,
            sessionID: sessionId,
            role,
            time: { created },
            agent: "build",
            model: { providerID: "vast", modelID: "qwen" },
        } as WithParts["info"],
        parts: [{ id: `${id}-part`, messageID: id, sessionID: sessionId, type: "text", text }],
    }
}

function plan(ids: string[], hash: string): BoundaryContextPlan {
    return {
        sessionId,
        rangeHash: hash,
        transcript: { relativePath: `legacy/${hash}.md`, content: "", messageIds: ids },
    } as BoundaryContextPlan
}

test("background description writes cannot erase a concurrent newly archived loop delta", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-deltas-concurrent-"))
    const older = native("older", "user", "Keep this correction", 1)
    const newer = native("newer", "assistant", "Agent continued without user input", 2)
    const first = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([older.info.id], "first"),
        originalMessages: [older, newer],
    })
    assert.ok(first.entry)
    const staleDescription = structuredClone(first.catalog)
    const [second, replay] = await Promise.all([
        archiveBoundaryDelta({
            directory,
            sessionId,
            plan: plan([older.info.id, newer.info.id], "second"),
            originalMessages: [older, newer],
        }),
        archiveBoundaryDelta({
            directory,
            sessionId,
            plan: plan([older.info.id, newer.info.id], "second"),
            originalMessages: [older, newer],
        }),
    ])
    assert.equal([second.entry, replay.entry].filter(Boolean).length, 1)
    staleDescription.entries[0].descriptionAttempts = 1
    await saveArchiveCatalog(directory, staleDescription)
    const latest = await loadArchiveCatalog(directory, sessionId)
    assert.equal(latest.entries.length, 2)
    assert.equal(latest.nextSequence, 3)
    assert.equal(latest.entries[0].descriptionAttempts, 1)
    assert.match(
        await readArchiveEntry(directory, latest, latest.entries[1].id),
        /Agent continued without user input/,
    )
})

test("successive compactions archive exact new native messages and stable replay writes nothing", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-deltas-"))
    const oldUser = native("msg-old-user", "user", "Exact first instruction", 1)
    const tool = native("msg-tool", "assistant", "", 2)
    tool.parts = [
        {
            id: "prt-tool",
            messageID: "msg-tool",
            sessionID: sessionId,
            type: "tool",
            tool: "bash",
            callID: "call1",
            state: { status: "completed", input: { command: "pwd" }, output: "exact tool output" },
        } as WithParts["parts"][number],
    ]
    const reasoning = native("msg-reasoning", "assistant", "An answer", 3)
    reasoning.parts.push({
        id: "prt-reasoning",
        messageID: reasoning.info.id,
        sessionID: sessionId,
        type: "reasoning",
        text: "private reasoning source",
    } as WithParts["parts"][number])
    const currentUser = native("msg-user-2", "user", "Updated instruction", 4)
    const first = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([oldUser.info.id, tool.info.id], "first"),
        originalMessages: [oldUser, tool, reasoning, currentUser],
    })
    assert.ok(first.entry)
    assert.equal(first.entry.sequence, 1)
    assert.equal(first.entry.status, "pending")
    assert.deepEqual(
        JSON.parse(
            (await readArchiveEntry(directory, first.catalog, first.entry.id))
                .split("```json\n")[1]
                .split("\n```")[0],
        ),
        [oldUser, tool],
    )

    const replay = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([oldUser.info.id, tool.info.id], "first"),
        originalMessages: [oldUser, tool, reasoning, currentUser],
    })
    assert.equal(replay.entry, null)
    assert.equal(replay.catalog.nextSequence, 2)

    const second = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([oldUser.info.id, tool.info.id, reasoning.info.id], "second"),
        originalMessages: [oldUser, tool, reasoning, currentUser],
    })
    assert.ok(second.entry)
    assert.equal(second.entry.sequence, 2)
    assert.deepEqual(Object.keys(second.entry.fingerprints), [reasoning.info.id])
    assert.ok(
        (await readArchiveEntry(directory, second.catalog, second.entry.id)).includes(
            "private reasoning source",
        ),
    )
    const checkpoint =
        "## Decisions\n- Work advanced.\n## Files & Symbols\n- src/app.ts\n## Errors (verbatim)\n- (none)\n## What failed and why\n- (none)\n## Constraints\n- Current task.\n## Next step\n- Continue."
    const turns = openCodeCodec.encode([oldUser, tool, reasoning, currentUser])
    assert.match(
        retainArchivedUserText(checkpoint, turns, turns.length, first.catalog, undefined, [oldUser, tool, reasoning, currentUser]),
        /Exact first instruction/,
    )
    const retired = retainArchivedUserText(checkpoint, turns, turns.length, second.catalog, 1, [oldUser, tool, reasoning, currentUser])
    assert.doesNotMatch(retired, /Exact first instruction/)
    assert.match(retired, /Updated instruction/)
    assert.equal(
        readFileSync(join(directory, ".opencode/better-compact/.gitignore"), "utf8"),
        "*\n!.gitignore\n",
    )
    assert.equal((await loadArchiveCatalog(directory, sessionId)).entries.length, 2)

    const changed = structuredClone(tool)
    const changedPart = changed.parts[0]
    if (changedPart?.type !== "tool" || changedPart.state.status !== "completed")
        throw new Error("fixture")
    changedPart.state.output = "updated tool output"
    const third = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([oldUser.info.id, changed.info.id, reasoning.info.id], "third"),
        originalMessages: [oldUser, changed, reasoning, currentUser],
    })
    assert.ok(third.entry)
    assert.deepEqual(Object.keys(third.entry.fingerprints), [changed.info.id])
    assert.equal(uncoveredMessages([oldUser, changed, reasoning], third.catalog).length, 0)

    const revisedUser = structuredClone(oldUser)
    const revisedPart = revisedUser.parts[0]
    assert.equal(revisedPart.type, "text")
    if (revisedPart.type === "text") revisedPart.text = "Exact first instruction: new violet correction"
    const revisedDelta = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([oldUser.info.id, tool.info.id, reasoning.info.id], "revised-user"),
        originalMessages: [revisedUser, changed, reasoning, currentUser],
    })
    assert.equal(revisedDelta.entry?.sequence, 4)
    assert.deepEqual(Object.keys(revisedDelta.entry!.fingerprints), [oldUser.info.id])
    const revisedLive = retainArchivedUserText(
        checkpoint,
        openCodeCodec.encode([revisedUser, changed, reasoning, currentUser]),
        turns.length,
        revisedDelta.catalog,
        1,
        [revisedUser, changed, reasoning, currentUser],
    )
    assert.match(revisedLive, /Exact first instruction: new violet correction/)
})

test("a split assistant message remains exact when a later completed part revises its payload", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-split-delta-"))
    const assistant = native("msg-split-assistant", "assistant", "Initial analysis", 1)
    assistant.parts.push({
        id: "prt-split-tool",
        messageID: assistant.info.id,
        sessionID: sessionId,
        type: "tool",
        tool: "bash",
        callID: "call-split",
        state: { status: "pending", input: { command: "pwd" }, raw: "{}" },
    } as WithParts["parts"][number])
    const first = await archiveBoundaryDelta({
        directory,
        sessionId,
        // A partial item boundary still names the owning native message.
        plan: plan([assistant.info.id], "split-first"),
        originalMessages: [assistant],
    })
    assert.ok(first.entry)
    const completed = structuredClone(assistant)
    completed.parts[1] = {
        ...completed.parts[1],
        state: {
            status: "completed",
            input: { command: "pwd" },
            output: "/project",
            title: "bash",
            metadata: {},
            time: { start: 1, end: 2 },
        },
    } as WithParts["parts"][number]
    const second = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([assistant.info.id], "split-second"),
        originalMessages: [completed],
    })
    assert.ok(second.entry)
    assert.notEqual(second.entry.checksum, first.entry.checksum)
    assert.deepEqual(Object.keys(second.entry.fingerprints), [assistant.info.id])
    const parse = async (id: string) =>
        JSON.parse(
            (await readArchiveEntry(directory, second.catalog, id))
                .split("```json\n")[1]
                .split("\n```")[0],
        ) as WithParts[]
    assert.deepEqual(await parse(first.entry.id), [assistant])
    assert.deepEqual(await parse(second.entry.id), [completed])
    const replay = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([assistant.info.id], "split-second"),
        originalMessages: [completed],
    })
    assert.equal(replay.entry, null)
    assert.equal(replay.catalog.entries.length, 2)
})

test("a failed plan save after delta publication leaves a recoverable single archive", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-plan-save-crash-"))
    const messages = [
        native("u-old", "user", "Preserve the active instruction", 1),
        native("a-old", "assistant", "Historical work ".repeat(1_000), 2),
        native("u-middle", "user", "Follow the latest decision", 3),
        native("a-middle", "assistant", "Recent work", 4),
        native("u-current", "user", "Continue with the current action", 5),
    ]
    let saved: PlanSnapshot | null = null
    let failSave = true
    const engine = createEngine(openCodeSpec, {
        transcripts: {
            citablePath: () => "legacy/plan-crash.md",
            write: async () => ({}),
        },
        plans: {
            load: () => saved,
            save: (_session, snapshot) => {
                if (failSave) throw new Error("simulated interrupted plan save")
                saved = snapshot
            },
        },
        archive: async (boundary) => {
            await archiveBoundaryDelta({
                directory,
                sessionId,
                plan: boundary,
                originalMessages: messages,
            })
        },
        logger: new Logger(false),
    })
    const request = {
        sessionKey: sessionId,
        turns: openCodeCodec.encode(messages),
        contextLimit: 10_000,
        force: true,
    }
    await assert.rejects(engine.process(request), /interrupted plan save/)
    const first = await loadArchiveCatalog(directory, sessionId)
    assert.equal(first.entries.length, 1)
    assert.equal(saved, null)
    assert.match(
        await readArchiveEntry(directory, first, first.entries[0].id),
        /Preserve the active instruction/,
    )

    failSave = false
    const result = await engine.process(request)
    assert.equal(result.outcome, "planned")
    assert.ok(saved)
    const recovered = await loadArchiveCatalog(directory, sessionId)
    assert.deepEqual(
        recovered.entries.map((entry) => entry.id),
        first.entries.map((entry) => entry.id),
    )
    assert.equal(recovered.nextSequence, 2)
})

test("a crash after writing a delta but before publishing its catalog reuses the same verified archive", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-orphan-delta-"))
    const source = native("msg-orphan", "user", "Retain this exact intent", 1)
    const first = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([source.info.id], "first"),
        originalMessages: [source],
    })
    assert.ok(first.entry)
    rmSync(archiveCatalogPath(directory, sessionId))
    const recovered = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([source.info.id], "first"),
        originalMessages: [source],
    })
    assert.equal(recovered.entry?.id, first.entry.id)
    assert.equal(recovered.catalog.entries.length, 1)
    assert.equal(recovered.catalog.nextSequence, 2)
    const archive = join(directory, first.entry.relativePath)
    rmSync(archiveCatalogPath(directory, sessionId))
    writeFileSync(archive, "tampered", "utf8")
    await assert.rejects(
        archiveBoundaryDelta({
            directory,
            sessionId,
            plan: plan([source.info.id], "first"),
            originalMessages: [source],
        }),
        /collision or modified orphan/,
    )
})

test("private archives fail closed for modified bytes and symlinked reads", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-delta-read-"))
    const message = native("msg-1", "user", "private content", 1)
    const result = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([message.info.id], "range"),
        originalMessages: [message],
    })
    assert.ok(result.entry)
    const full = join(directory, result.entry.relativePath)
    const link = join(dirname(full), "catalog-symlink.md")
    symlinkSync(full, link)
    await assert.rejects(readPrivateFile(link, directory), /symlinked Better Compact path/)
    const altered = { ...result.catalog, entries: [{ ...result.entry, checksum: "00" }] }
    await assert.rejects(
        readArchiveEntry(directory, altered, result.entry.id),
        /missing_or_modified_file/,
    )
    await assert.rejects(loadArchiveCatalog(directory, "../escape"), /Invalid archive session ID/)
})

test("seven-day expiry removes private payloads but preserves stable session tombstones", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-expiry-"))
    const source = native("msg-expiry", "user", "Old exact instruction", 1)
    const archived = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: plan([source.info.id], "expired"),
        originalMessages: [source],
    })
    const entry = archived.entry!
    await archiveOversizedSummary(directory, archived.catalog, entry.id, "Private rejected summary")
    const summaryPath = (await loadArchiveCatalog(directory, sessionId)).entries[0]
        .oversizedSummaryPath!
    const created = Date.parse(entry.createdAt)
    await expireArchives(directory, sessionId, created + ARCHIVE_RETENTION_MS - 1)
    assert.ok(existsSync(join(directory, entry.relativePath)))
    await expireArchives(directory, sessionId, created + ARCHIVE_RETENTION_MS)
    const catalog = await loadArchiveCatalog(directory, sessionId)
    assert.equal(catalog.entries[0].status, "expired")
    assert.equal(catalog.nextSequence, 2)
    assert.equal(uncoveredMessages([source], catalog).length, 0)
    assert.ok(!existsSync(join(directory, entry.relativePath)))
    assert.ok(!existsSync(join(directory, summaryPath)))
    await assert.rejects(readArchiveEntry(directory, catalog, entry.id), /expired_archive/)
    await assert.rejects(readArchivedSummary(directory, catalog, entry.id), /expired_archive/)
    await expireArchives(directory, sessionId, created + ARCHIVE_RETENTION_MS + 1)
    assert.equal((await loadArchiveCatalog(directory, sessionId)).entries.length, 1)
})

test("startup backfills only the opened project without losing nested-project pending archives", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bc-cwd-projects-"))
    const projects = [cwd, join(cwd, "group", "project-two")]
    for (const directory of projects) {
        mkdirSync(directory, { recursive: true })
        const source = native("msg-pending", "user", "Project-specific historical requirement", 1)
        await archiveBoundaryDelta({
            directory,
            sessionId,
            plan: plan([source.info.id], "first"),
            originalMessages: [source],
        })
    }
    assert.deepEqual(await projectArchiveRoots(cwd), [cwd])
    assert.deepEqual(await projectArchiveRoots(projects[1]), [projects[1]])
    const visited: string[] = []
    const sdk = {
        provider: {
            list: async () => ({
                data: {
                    all: [
                        { id: "openai", models: { "gpt-6-luna": { limit: { context: 100_000 } } } },
                    ],
                },
            }),
        },
        session: {
            create: async ({ query }: any) => {
                visited.push(query.directory)
                return { data: { id: `scratch-${visited.length}` } }
            },
            prompt: async ({ body, query }: any) => {
                assert.ok(projects.includes(query.directory))
                assert.equal(body.variant, undefined)
                return {
                    data: {
                        parts: [
                            {
                                type: "text",
                                text: JSON.stringify({
                                    description:
                                        "Historical project requirement and implementation evidence.",
                                }),
                            },
                        ],
                    },
                }
            },
            delete: async ({ query }: any) => {
                assert.ok(projects.includes(query.directory))
                return { data: true }
            },
        },
    }
    const logger = new Logger(false)
    await startArchiveDescriptionBackfill({
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        directory: cwd,
        summaryModel: "openai/gpt-6-luna",
    })
    assert.deepEqual(visited, [cwd])
    assert.equal((await loadArchiveCatalog(projects[1], sessionId)).entries[0].status, "pending")
    await startArchiveDescriptionBackfill({
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        directory: projects[1],
        summaryModel: "openai/gpt-6-luna",
    })
    assert.deepEqual(new Set(visited), new Set(projects))
    for (const directory of projects) {
        const catalog = await loadArchiveCatalog(directory, sessionId)
        assert.equal(catalog.entries[0].status, "ready")
    }
})

test("startup discovery refuses symlinked private archive roots", async () => {
    const cwd = mkdtempSync(join(tmpdir(), "bc-startup-link-"))
    const external = mkdtempSync(join(tmpdir(), "bc-startup-external-"))
    const source = native("msg-foreign", "user", "Other project's wording", 1)
    await archiveBoundaryDelta({
        directory: external,
        sessionId,
        plan: plan([source.info.id], "external"),
        originalMessages: [source],
    })
    symlinkSync(join(external, ".opencode"), join(cwd, ".opencode"))
    await assert.rejects(projectArchiveRoots(cwd), /Unsafe archive startup directory/)
})
