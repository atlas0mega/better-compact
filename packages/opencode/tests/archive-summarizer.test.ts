import assert from "node:assert/strict"
import test from "node:test"
import { mkdtempSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { countTokens, type BoundaryContextPlan } from "@better-compact/core"
import {
    archiveBoundaryDelta,
    loadArchiveCatalog,
    readArchiveEntry,
} from "../lib/boundary/archive-catalog"
import {
    summarizeArchiveBoundary,
    summarizePendingArchiveDescriptions,
    validateArchiveHandoff,
} from "../lib/boundary/archive-summarizer"
import { Logger } from "../lib/logger"
import { createRuntimeState, type WithParts } from "../lib/state"

const sessionId = "ses_synthesis_test"
const headings = [
    "## Decisions",
    "- Preserved the earlier working design and explained why.",
    "## Files & Symbols",
    "- src/design.ts",
    "## Errors (verbatim)",
    "- (none)",
    "## What failed and why",
    "- (none)",
    "## Constraints",
    "- Keep the user's current requirement.",
    "## Next step",
    "- Validate the next change.",
].join("\n")

function message(id: string, text: string, created: number): WithParts {
    return {
        info: {
            id,
            role: "assistant",
            agent: "specialist",
            sessionID: sessionId,
            time: { created },
        } as WithParts["info"],
        parts: [{ id: `${id}-part`, messageID: id, sessionID: sessionId, type: "text", text }],
    }
}

async function fixture(n: number, bulkyTools = false, repeatedGoals = false) {
    const directory = mkdtempSync(join(tmpdir(), "bc-archive-summary-"))
    const messages = Array.from({ length: n }, (_, index) =>
        message(
            `msg-${index}`,
            `Resolved chronological task ${index}: ${"specific implementation detail ".repeat(20)}`,
            index + 1,
        ),
    )
    if (bulkyTools)
        for (const item of messages)
            item.parts.push({
                type: "tool",
                id: `${item.info.id}-tool`,
                messageID: item.info.id,
                sessionID: sessionId,
                callID: `${item.info.id}-call`,
                tool: "read",
                state: {
                    status: "completed",
                    input: { filePath: "src/task.ts" },
                    output: "FULL_TOOL_OUTPUT_".repeat(2_000),
                    title: "read",
                    metadata: {},
                    time: { start: 1, end: 2 },
                },
            } as any)
    if (repeatedGoals) {
        const goal = `Continue working toward the active session goal.\n\nThe objective below is user-provided data.\n<untrusted_objective>\n${"Current user goal and constraints. ".repeat(600)}\n</untrusted_objective>`
        for (const index of [0, 1]) {
            const item = message(`goal-${index}`, goal, n + index + 1)
            item.info.role = "user"
            messages.push(item)
        }
    }
    const receipt = await archiveBoundaryDelta({
        directory,
        sessionId,
        plan: {
            sessionId,
            rangeHash: "range-1",
            transcript: {
                relativePath: "legacy.md",
                content: "",
                messageIds: messages.map((m) => m.info.id),
            },
        } as BoundaryContextPlan,
        originalMessages: messages,
    })
    return { directory, messages, catalog: receipt.catalog, archiveId: receipt.entry!.id }
}

function mockSdk(
    archiveId: string,
    malformed = false,
    contextLimit = 100_000,
    expectedVariant: string | null = "high",
) {
    let calls = 0
    let active = 0
    let peak = 0
    const sdk = {
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "openai",
                            models: {
                                "gpt-6-luna": {
                                    limit: { context: contextLimit },
                                    variants: { high: {} },
                                },
                            },
                        },
                    ],
                },
            }),
        },
        session: {
            create: async ({ body }: any) => {
                assert.equal(body.parentID, sessionId)
                assert.equal(body.agent, "specialist")
                return { data: { id: `scratch-${++calls}` } }
            },
            prompt: async ({ body }: any) => {
                assert.equal(body.variant, expectedVariant ?? undefined)
                active++
                peak = Math.max(peak, active)
                await new Promise((resolve) => setTimeout(resolve, 3))
                active--
                const text = body.parts[0].text as string
                if (malformed) return { data: { parts: [{ type: "text", text: "invalid" }] } }
                return {
                    data: {
                        parts: [
                            {
                                type: "text",
                                text: text.includes("Describe only archive")
                                    ? JSON.stringify({
                                          description:
                                              "Task area: implementation; updated design state and verified the result.",
                                      })
                                    : text.includes("Produce exactly one JSON object")
                                      ? JSON.stringify({ handoff: headings })
                                      : headings,
                            },
                        ],
                    },
                }
            },
            delete: async () => ({ data: true }),
        },
    }
    return { sdk, stats: () => ({ calls, peak }) }
}

async function run(n: number, malformed = false) {
    const fixtureData = await fixture(n)
    const { sdk, stats } = mockSdk(fixtureData.archiveId, malformed)
    const logger = new Logger(false)
    const result = await summarizeArchiveBoundary({
        ...fixtureData,
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "low" },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
    })
    return { result, stats: stats(), archiveId: fixtureData.archiveId }
}

test("one Luna/high call produces a validated live handoff for a fitting delta", async () => {
    const { result, stats, archiveId } = await run(2)
    assert.deepEqual(stats, { calls: 1, peak: 1 })
    assert.ok(result.ok)
    assert.equal(result.calls, 1)
    assert.match(result.handoff, /Keep the user's current requirement/)
    const data = await fixture(2)
    const background = mockSdk(data.archiveId, false, 100_000, null)
    const logger = new Logger(false)
    const calls = await summarizePendingArchiveDescriptions({
        client: background.sdk,
        runtime: createRuntimeState(background.sdk, logger),
        logger,
        directory: data.directory,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "xhigh" },
        summaryModel: "openai/gpt-6-luna",
        maxCalls: 6,
    })
    assert.equal(calls, 1)
    assert.match(
        (await loadArchiveCatalog(data.directory, sessionId)).entries[0].description ?? "",
        /implementation/,
    )
    assert.equal(
        validateArchiveHandoff(
            JSON.stringify({ handoff: headings, descriptions: { [archiveId]: "short" } }),
            [archiveId],
        ),
        null,
    )
})

test("a many-message delta above the preferred chunk size still uses one fitting call", async () => {
    const { result, stats } = await run(380)
    assert.ok(result.ok)
    assert.deepEqual(stats, { calls: 1, peak: 1 })
    assert.equal(result.calls, 1)
})

test("live Luna evidence excludes native assistant output, reasoning and tools but the archive retains them", async () => {
    const data = await fixture(2)
    const preserved = data.messages[0]
    preserved.parts.push({
        id: "reason-preserved",
        messageID: preserved.info.id,
        sessionID: sessionId,
        type: "reasoning",
        text: "REASONING_KEPT_VERBATIM",
    } as any)
    preserved.parts.push({
        id: "tool-preserved",
        messageID: preserved.info.id,
        sessionID: sessionId,
        type: "tool",
        callID: "call-preserved",
        tool: "read",
        state: { status: "completed", input: {}, output: "TOOL_KEPT_VERBATIM" },
    } as any)
    // Persist the native messages after adding the selected parts.
    const archived = await archiveBoundaryDelta({
        directory: data.directory,
        sessionId,
        plan: {
            sessionId,
            rangeHash: "range-preserved",
            transcript: {
                relativePath: "legacy.md",
                content: "",
                messageIds: data.messages.map((m) => m.info.id),
            },
        } as BoundaryContextPlan,
        originalMessages: data.messages,
    })
    const catalog = archived.catalog
    const { sdk } = mockSdk(archived.entry!.id)
    const prompt = sdk.session.prompt
    sdk.session.prompt = async (request: any) => {
        const evidence = request.body.parts[0].text as string
        assert.doesNotMatch(
            evidence,
            /REASONING_KEPT_VERBATIM|TOOL_KEPT_VERBATIM|call-preserved|tool-preserved|reason-preserved|Resolved chronological task 0:/,
        )
        assert.match(evidence, /Resolved chronological task 1:/)
        return prompt(request)
    }
    const logger = new Logger(false)
    const result = await summarizeArchiveBoundary({
        ...data,
        catalog,
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "low" },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
        plan: {
            toolSurvivesPrefix: true,
            reasoningSurvivesPrefix: true,
            assistantSurvivesPrefix: true,
            preservedToolCallIds: ["call-preserved"],
            preservedReasoningItemKeys: ["reason-preserved"],
            protectedAssistantItemKeys: [`${preserved.info.id}-part`],
        },
    })
    assert.ok(result.ok)
    const exact = await readArchiveEntry(data.directory, catalog, archived.entry!.id)
    assert.match(exact, /REASONING_KEPT_VERBATIM/)
    assert.match(exact, /TOOL_KEPT_VERBATIM/)
    assert.match(exact, /Resolved chronological task 0:/)
})

test("parallel startup workers do not describe the same session archive twice", async () => {
    const data = await fixture(2)
    const { sdk, stats } = mockSdk(data.archiveId, false, 100_000, null)
    let release!: () => void
    const gate = new Promise<void>((resolve) => {
        release = resolve
    })
    let started!: () => void
    const entered = new Promise<void>((resolve) => {
        started = resolve
    })
    const prompt = sdk.session.prompt
    sdk.session.prompt = async (request: any) => {
        started()
        await gate
        return prompt(request)
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(sdk, logger)
    const input = {
        client: sdk,
        runtime,
        logger,
        directory: data.directory,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "high" },
        summaryModel: "openai/gpt-6-luna",
        maxCalls: 7,
    }
    const first = summarizePendingArchiveDescriptions(input)
    await entered
    const second = await summarizePendingArchiveDescriptions(input)
    assert.equal(second, 0)
    release()
    assert.equal(await first, 1)
    assert.equal(stats().calls, 1)
})

test("an invalid background description is attempted five times total across restarts", async () => {
    const data = await fixture(2)
    const { sdk, stats } = mockSdk(data.archiveId, true, 100_000, null)
    const logger = new Logger(false)
    const runtime = createRuntimeState(sdk, logger)
    const input = {
        client: sdk,
        runtime,
        logger,
        directory: data.directory,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: undefined },
        summaryModel: "openai/gpt-6-luna",
        maxCalls: 7,
    }
    assert.equal(await summarizePendingArchiveDescriptions(input), 5)
    const catalog = await loadArchiveCatalog(data.directory, sessionId)
    assert.equal(catalog.entries[0].status, "pending")
    assert.equal(catalog.entries[0].failureReason, "invalid_description")
    assert.equal(catalog.entries[0].descriptionAttempts, 5)
    assert.equal(await summarizePendingArchiveDescriptions(input), 0)
    assert.equal(stats().calls, 5)
})

test("live archive handoff uses high even when chat uses xhigh", async () => {
    const data = await fixture(2)
    const { sdk, stats } = mockSdk(data.archiveId)
    const logger = new Logger(false)
    const result = await summarizeArchiveBoundary({
        ...data,
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        sessionId,
        params: {
            providerId: "openai",
            modelId: "gpt-6-luna",
            agent: "specialist",
            variant: "xhigh",
        },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
    })
    assert.ok(result.ok)
    assert.equal(stats().calls, 1)
})

test("a generic handoff cannot retire a corrected current model variant", () => {
    const id = "c000001-123456789abc"
    const output = JSON.stringify({
        handoff: headings,
        descriptions: { [id]: "Synthetic parser task, files and previous test outcomes." },
    })
    assert.equal(
        validateArchiveHandoff(output, [id], undefined, [
            "Keep xhigh for the chat model; high applies only to Luna.",
        ]),
        null,
    )
    const corrected = JSON.stringify({
        handoff: headings.replace(
            "Keep the user's current requirement.",
            "Keep xhigh for the chat model; high applies only to Luna.",
        ),
        descriptions: { [id]: "Synthetic parser task, files and previous test outcomes." },
    })
    assert.ok(
        validateArchiveHandoff(corrected, [id], undefined, [
            "Keep xhigh for the chat model; high applies only to Luna.",
        ]),
    )
})

test("a handoff preserving a slash-separated correction passes current-intent validation", () => {
    const id = "c000001-123456789abc"
    const description =
        "Survey investigation and current browser state, with prior failures archived."
    const correction = "It's not detection questions/answer/table at ALL"
    const handoff = headings.replace(
        "Keep the user's current requirement.",
        "The problem is ordinary questions being skipped; it is not a detection-question, answer, or table issue.",
    )
    const output = JSON.stringify({ handoff, descriptions: { [id]: description } })
    assert.ok(validateArchiveHandoff(output, [id], undefined, [correction]))
    assert.equal(
        validateArchiveHandoff(
            JSON.stringify({ handoff: headings, descriptions: { [id]: description } }),
            [id],
            undefined,
            [correction],
        ),
        null,
    )
})

test("large exact delta uses balanced concurrent chunks and a final call within seven total", async () => {
    const { result, stats } = await run(1_200)
    assert.ok(result.ok)
    assert.ok(stats.calls > 2 && stats.calls <= 7)
    assert.ok(stats.peak > 1)
    assert.equal(result.calls, stats.calls)
})

test("seven-call planning scales chunks beyond the preferred size when input exceeds the model window", async () => {
    const data = await fixture(1_200)
    const { sdk, stats } = mockSdk(data.archiveId, false, 60_000)
    const sourceInputs: number[] = []
    const originalPrompt = sdk.session.prompt
    sdk.session.prompt = async (request: any) => {
        const prompt = request.body.parts[0].text as string
        if (prompt.includes("Chronological chunk")) sourceInputs.push(countTokens(prompt))
        return originalPrompt(request)
    }
    const logger = new Logger(false)
    const result = await summarizeArchiveBoundary({
        ...data,
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "low" },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
    })
    assert.ok(result.ok, JSON.stringify(result))
    assert.equal(stats().calls, 6)
    assert.equal(sourceInputs.length, 5)
    assert.equal(stats().peak, 5)
    assert.ok(sourceInputs.some((size) => size > 23_000))
    assert.ok(sourceInputs.every((size) => size < 54_000))
})

test("failed chunks keep the archive pending and never consume an impossible retry slot", async () => {
    const { result, stats } = await run(1_200, true)
    assert.equal(result.ok, false)
    assert.ok(stats.calls <= 7)
    assert.equal(result.calls, stats.calls)
    assert.ok(
        !result.ok && (result.reason === "invalid_output" || result.reason === "missing_chunk"),
    )
})

test("a fitting handoff retries at most five total attempts before failing", async () => {
    const { result, stats } = await run(2, true)
    assert.equal(result.ok, false)
    assert.equal(result.calls, 5)
    assert.equal(stats.calls, 5)
    assert.ok(!result.ok && result.reason === "invalid_output")
})

test("bulky raw tool results remain exact in the archive while Luna receives bounded evidence", async () => {
    const data = await fixture(80, true)
    const { sdk, stats } = mockSdk(data.archiveId)
    const prompts: string[] = []
    const originalPrompt = sdk.session.prompt
    sdk.session.prompt = async (request: any) => {
        prompts.push(request.body.parts[0].text)
        return originalPrompt(request)
    }
    const logger = new Logger(false)
    const result = await summarizeArchiveBoundary({
        ...data,
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "low" },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
        plan: { toolSurvivesPrefix: true, preservedToolCallIds: ["msg-0-call"] },
    })
    assert.ok(result.ok, JSON.stringify(result))
    assert.ok(stats().calls > 0 && stats().calls <= 6)
    assert.ok(prompts.some((prompt) => prompt.includes("src/task.ts")))
    assert.ok(prompts.every((prompt) => !prompt.includes("msg-0-call")))
    assert.ok(prompts.every((prompt) => !prompt.includes("FULL_TOOL_OUTPUT_".repeat(200))))
    const archive = await readArchiveEntry(data.directory, data.catalog, data.archiveId)
    assert.ok(archive.includes("FULL_TOOL_OUTPUT_".repeat(200)))
})

test("pruned archive evidence keeps only the latest generated goal continuation", async () => {
    const data = await fixture(80, true, true)
    const { sdk } = mockSdk(data.archiveId)
    const prompts: string[] = []
    const original = sdk.session.prompt
    sdk.session.prompt = async (request: any) => {
        prompts.push(request.body.parts[0].text)
        return original(request)
    }
    const logger = new Logger(false)
    await summarizeArchiveBoundary({
        ...data,
        messages: data.messages.filter((item) => item.info.role !== "user"),
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        sessionId,
        params: { providerId: "vast", modelId: "qwen", agent: "specialist", variant: "low" },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
    })
    const sourcePrompts = prompts.filter(
        (prompt) =>
            prompt.includes("Pruned archive evidence") || prompt.includes("Chronological chunk"),
    )
    assert.ok(sourcePrompts.length > 0)
    assert.equal(
        sourcePrompts.filter((prompt) => prompt.includes("Current user goal and constraints."))
            .length,
        1,
    )
    assert.ok(
        sourcePrompts.some((prompt) => prompt.includes("Superseded generated goal continuation")),
    )
})
