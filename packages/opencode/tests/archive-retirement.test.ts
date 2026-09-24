import assert from "node:assert/strict"
import test from "node:test"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import type { PluginConfig } from "../lib/config"
import { Logger } from "../lib/logger"
import { createSessionState, type WithParts } from "../lib/state"
import { processBoundaryTransform, retainArchivedUserText } from "../lib/boundary/engine"
import { openCodeCodec } from "../lib/codec"
import {
    loadArchiveCatalog,
    readArchiveEntry,
    saveArchiveCatalog,
} from "../lib/boundary/archive-catalog"

const sessionId = "ses_two_boundary_intent"

function message(id: string, role: "user" | "assistant", text: string, created: number): WithParts {
    return {
        info: {
            id,
            sessionID: sessionId,
            role,
            agent: "build",
            time: { created },
            model: { providerID: "openai", modelID: "gpt-6-sol" },
        } as WithParts["info"],
        parts: [{ id: `prt-${id}`, messageID: id, sessionID: sessionId, type: "text", text }],
    }
}

function config(): PluginConfig {
    return {
        enabled: true,
        autoUpdate: false,
        debug: false,
        commands: { enabled: true },
        compaction: {
            automatic: true,
            preset: "custom",
            summaryEffort: "high",
            custom: {
                triggerPercent: 1,
                targetPercent: 1,
                recentToolTokens: 0,
                summarizerConcurrency: 5,
                prefixSummary: true,
                collapsePercent: 45,
            },
        },
        experimental: { allowSubAgents: true },
        compress: { permission: "allow" },
    }
}

test("a validated handoff keeps first-boundary wording until older descriptions are ready", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-intent-aging-"))
    const logger = new Logger(false)
    const state = createSessionState(sessionId)
    state.modelContextLimit = 20_000
    const original = [
        message("u-1", "user", "Use the original blue policy for parser migration.", 1),
        message(
            "a-1",
            "assistant",
            `Blue policy implementation notes in src/parser.ts: ${"fixed migration case. ".repeat(2_000)}`,
            2,
        ),
        message("u-2", "user", "Check the parser migration tests.", 3),
        message("a-2", "assistant", "The migration tests showed an old failure.", 4),
        message("u-3", "user", "Correction: use red policy; blue was superseded.", 5),
    ]
    const first = structuredClone(original)
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: first,
        summariesAllowed: false,
    })
    const afterFirst = await loadArchiveCatalog(directory, sessionId)
    assert.equal(afterFirst.entries.length, 1)
    assert.equal(afterFirst.retirementThrough, undefined)
    assert.match(JSON.stringify(first), /Use the original blue policy for parser migration/)
    assert.deepEqual(
        first.find((entry) => entry.info.id === "u-3"),
        original.find((entry) => entry.info.id === "u-3"),
        "the latest, non-archived user correction stays native and byte-exact",
    )

    let settingsOnlyCalls = 0
    const settingsOnly = config()
    settingsOnly.compaction.custom!.targetPercent = 2
    await processBoundaryTransform({
        state,
        logger,
        config: settingsOnly,
        directory,
        messages: structuredClone(original),
        summariesAllowed: true,
        summarizeArchive: async () => {
            settingsOnlyCalls++
            return { ok: false, calls: 1, reason: "invalid_output" }
        },
    })
    assert.equal(settingsOnlyCalls, 0, "settings-only rebuild must not retry pending Luna work")
    assert.equal((await loadArchiveCatalog(directory, sessionId)).entries.length, 1)

    const second = [
        ...structuredClone(original),
        message(
            "a-3",
            "assistant",
            `Red migration state: ${"src/parser.ts tests checked. ".repeat(800)}`,
            6,
        ),
        message("u-4", "user", "Keep red policy and validate parser migration tests next.", 7),
    ]
    const handoff = [
        "## Decisions",
        "- Red supersedes blue for parser migration.",
        "## Files & Symbols",
        "- src/parser.ts",
        "## Errors (verbatim)",
        "- Prior migration test failed.",
        "## What failed and why",
        "- The original blue policy did not satisfy the migration tests.",
        "## Constraints",
        "- Use red policy; blue was superseded. Keep the parser migration contract.",
        "## Next step",
        "- Validate parser migration tests next.",
    ].join("\n")
    let calls = 0
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: second,
        summariesAllowed: true,
        summarizeArchive: async (_plan, _turns, catalog) => {
            calls++
            return {
                ok: true,
                calls: 1,
                handoff,
                descriptions: Object.fromEntries(
                    catalog.entries
                        .filter((e) => e.status === "pending")
                        .map((e) => [
                            e.id,
                            `Parser migration ${e.sequence}: blue policy changed to red, tests next.`,
                        ]),
                ),
            }
        },
    })
    const afterSecond = await loadArchiveCatalog(directory, sessionId)
    assert.equal(afterSecond.entries.length, 2)
    assert.equal(calls, 1)
    assert.equal(afterSecond.retirementThrough, undefined)
    assert.ok(
        afterSecond.entries.every((e) => e.status === "pending"),
        "descriptions run separately in the background",
    )
    assert.equal(afterSecond.checkpoint, handoff)
    assert.match(JSON.stringify(second), /Use the original blue policy for parser migration/)
    assert.match(JSON.stringify(second), /red policy/i)
    assert.deepEqual(
        second.find((entry) => entry.info.id === "u-4"),
        message("u-4", "user", "Keep red policy and validate parser migration tests next.", 7),
        "the new, non-archived user instruction remains native after retirement",
    )
    assert.match(
        await readArchiveEntry(directory, afterSecond, afterSecond.entries[0].id),
        /Use the original blue policy for parser migration/,
    )
})

test("a first-boundary handoff that already quotes a user correction does not count it twice", () => {
    const correction = "Correction: use red policy; blue was superseded."
    const turns = openCodeCodec.encode([
        message("u-old", "user", correction, 1),
        message("a-old", "assistant", "A response", 2),
        message("u-current", "user", "Check the tests next.", 3),
    ])
    const handoff = [
        "## Decisions",
        "- Red policy is active.",
        "## Files & Symbols",
        "- src/parser.ts",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- Blue policy was superseded.",
        "## Constraints",
        `- ${correction}`,
        "## Next step",
        "- Check the tests next.",
    ].join("\n")
    const retained = retainArchivedUserText(handoff, turns, 2)
    assert.equal(retained.split(correction).length - 1, 1)
    assert.equal(retained, handoff)
})

test("an older catalog checkpoint alone cannot retire first-boundary wording", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-checkpoint-migration-"))
    const logger = new Logger(false)
    const state = createSessionState(sessionId)
    state.modelContextLimit = 20_000
    const original = [
        message("u-old", "user", "Exact legacy blue policy that was superseded.", 1),
        message("a-old", "assistant", `Historical old work ${"detail ".repeat(4_000)}`, 2),
        message("u-middle", "user", "Check migration tests next.", 3),
        message("a-middle", "assistant", "Work in src/parser.ts", 4),
        message("u-current", "user", "Use current red policy for migration.", 5),
    ]
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: structuredClone(original),
        summariesAllowed: false,
    })
    const before = await loadArchiveCatalog(directory, sessionId)
    assert.equal(before.entries.length, 1)
    const checkpoint = [
        "## Decisions",
        "- Red supersedes the old blue migration policy.",
        "## Files & Symbols",
        "- src/parser.ts",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- The legacy policy was superseded.",
        "## Constraints",
        "- Use the current red policy and check migration tests.",
        "## Next step",
        "- Check the migration tests in src/parser.ts.",
    ].join("\n")
    before.checkpoint = checkpoint
    before.validatedCheckpointId = before.entries[0].id
    await saveArchiveCatalog(directory, before)
    const migrated = structuredClone(original)
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: migrated,
        summariesAllowed: false,
    })
    const after = await loadArchiveCatalog(directory, sessionId)
    assert.equal(after.entries.length, 1)
    assert.equal(after.retirementThrough, undefined)
    assert.match(JSON.stringify(migrated), /Exact legacy blue policy that was superseded/)
    assert.match(JSON.stringify(migrated), /current red policy/i)
    assert.equal(state.boundary.activePlan?.retirementThrough, undefined)
    const replay = structuredClone(original)
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: replay,
        summariesAllowed: false,
    })
    assert.deepEqual(replay, migrated)
    assert.equal((await loadArchiveCatalog(directory, sessionId)).entries.length, 1)
})

test("round two retires only described older wording and keeps the newly archived correction", async () => {
    const directory = mkdtempSync(join(tmpdir(), "bc-round-two-ready-"))
    const logger = new Logger(false)
    const state = createSessionState(sessionId)
    state.modelContextLimit = 30_000
    const initial = [
        message("first-user", "user", "Old blue policy verbatim.", 1),
        message("first-work", "assistant", `Historical blue work ${"detail ".repeat(4_000)}`, 2),
        message("middle-user", "user", "Check the migration tests.", 3),
        message("middle-work", "assistant", "Reviewed src/parser.ts", 4),
        ...Array.from({ length: 6 }, (_, index) =>
            message(`later-work-${index}`, "assistant", `Checked parser stage ${index}.`, 5 + index),
        ),
        message("latest-user", "user", "Correction: use red policy instead of blue.", 11),
    ]
    const first = structuredClone(initial)
    const sections = (decision: string) =>
        [
            "## Decisions",
            `- ${decision}`,
            "## Files & Symbols",
            "- src/parser.ts",
            "## Errors (verbatim)",
            "- (none)",
            "## What failed and why",
            "- Blue is superseded.",
            "## Constraints",
            "- Use red policy instead of blue.",
            "## Next step",
            "- Validate the migration tests.",
        ].join("\n")
    let firstCheapProjection = 0
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: first,
        summariesAllowed: true,
        summarizeArchive: async (plan) => {
            firstCheapProjection = plan.afterPruneTokens
            assert.equal(plan.requiresCustomCompaction, false)
            return { ok: true, calls: 1, handoff: sections("Blue was replaced by red.") }
        },
    })
    const catalog = await loadArchiveCatalog(directory, sessionId)
    assert.equal(catalog.entries.length, 1)
    assert.equal(catalog.retirementThrough, undefined)
    assert.match(JSON.stringify(first), /Old blue policy verbatim/)
    assert.ok(catalog.checkpoint)
    assert.ok(state.boundary.activePlan?.requiresCustomCompaction)
    assert.ok(state.boundary.activePlan.afterPruneTokens < firstCheapProjection)

    catalog.entries[0].status = "ready"
    catalog.entries[0].description = "Blue parser work superseded by red; tests are next."
    await saveArchiveCatalog(directory, catalog)
    const advanced = [
        ...structuredClone(initial),
        message("new-work", "assistant", `Red parser migration ${"checked tests ".repeat(1_500)}`, 12),
        message("new-user", "user", "Keep the red correction and finish tests.", 13),
    ]
    let secondCheapProjection = 0
    await processBoundaryTransform({
        state,
        logger,
        config: config(),
        directory,
        messages: advanced,
        summariesAllowed: true,
        summarizeArchive: async (plan) => {
            secondCheapProjection = plan.afterPruneTokens
            assert.equal(plan.requiresCustomCompaction, false)
            return { ok: true, calls: 1, handoff: sections("Red policy is current.") }
        },
    })
    const second = await loadArchiveCatalog(directory, sessionId)
    assert.equal(second.entries.length, 2)
    assert.equal(second.retirementThrough, 1)
    assert.ok(state.boundary.activePlan?.requiresCustomCompaction)
    assert.ok(state.boundary.activePlan.afterPruneTokens < secondCheapProjection)
    assert.doesNotMatch(JSON.stringify(advanced), /Old blue policy verbatim/)
    assert.match(JSON.stringify(advanced), /Correction: use red policy instead of blue/)
    assert.deepEqual(
        advanced.find((entry) => entry.info.id === "new-user"),
        message("new-user", "user", "Keep the red correction and finish tests.", 13),
    )
    assert.match(await readArchiveEntry(directory, second, second.entries[0].id), /Old blue policy verbatim/)
})
