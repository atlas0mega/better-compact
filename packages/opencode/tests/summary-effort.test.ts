import assert from "node:assert/strict"
import test from "node:test"
import { resolveCompactionVariant, summarizeBoundaryJobs } from "../lib/boundary/summarizer"
import { Logger } from "../lib/logger"
import { createRuntimeState } from "../lib/state"

const params = { providerId: "local", modelId: "qwen", agent: undefined, variant: "high" }
const client = {
    provider: {
        list: async () => ({
            data: {
                all: [
                    {
                        id: "local",
                        models: { qwen: { variants: { low: {}, high: {}, xhigh: {} } } },
                    },
                ],
            },
        }),
    },
}

test("compaction reasoning uses the requested model variant", async () => {
    assert.equal(await resolveCompactionVariant(client, params, "low"), "low")
    assert.equal(await resolveCompactionVariant(client, params, "max"), "xhigh")
    assert.equal(params.variant, "high")
})

test("inherit, unsupported effort and provider errors preserve the active variant", async () => {
    assert.equal(await resolveCompactionVariant(client, params, "inherit"), "high")
    assert.equal(await resolveCompactionVariant(client, params, "medium"), "high")
    assert.equal(await resolveCompactionVariant({}, params, "low"), "high")
    assert.equal(
        await resolveCompactionVariant(client, { ...params, modelId: "other" }, "low"),
        "high",
    )
})

const job = {
    key: "run-1",
    rangeStartMessageId: "msg-start",
    rangeEndMessageId: "msg-end",
    transcriptRelativePath: "transcripts/reference.md",
    prompt: "Summarize the completed assistant work.",
}
const validSummary = [
    "## Decisions",
    "- Delivered the change.",
    "## Files & Symbols",
    "- src/app.ts",
    "## Errors (verbatim)",
    "- (none)",
    "## What failed and why",
    "- (none)",
    "## Constraints",
    "- Keep the output consistent.",
    "## Next step",
    "- Review the change.",
].join("\n")

test("scratch summaries use session-create model IDs and prompt with modelID", async () => {
    const calls: string[] = []
    const logger = new Logger(false)
    let runtime: ReturnType<typeof createRuntimeState>
    const sdk = {
        session: {
            create: async ({ body }: any) => {
                assert.deepEqual(body.model, { providerID: "local", id: "qwen" })
                assert.equal(body.parentID, "parent-session")
                calls.push("create")
                return { data: { id: "scratch-session" } }
            },
            prompt: async ({ path, body }: any) => {
                assert.equal(path.id, "scratch-session")
                assert.deepEqual(body.model, { providerID: "local", modelID: "qwen" })
                assert.equal(body.variant, "high")
                assert.equal(runtime.isScratch("scratch-session"), true)
                calls.push("prompt")
                return { data: { parts: [{ type: "text", text: validSummary }] } }
            },
            delete: async ({ path }: any) => {
                assert.equal(path.id, "scratch-session")
                calls.push("delete")
            },
        },
    }
    runtime = createRuntimeState(sdk, logger)
    const summaries = await summarizeBoundaryJobs({
        client: sdk,
        runtime,
        logger,
        parentSessionId: "parent-session",
        jobs: [job],
        params,
        summaryEffort: "inherit",
    })

    assert.equal(summaries[job.key], validSummary)
    assert.deepEqual(calls, ["create", "prompt", "delete"])
    assert.equal(runtime.isScratch("scratch-session"), false)
})

test("scratch summaries route to a chosen model with its own high reasoning variant", async () => {
    const logger = new Logger(false)
    const sdk = {
        provider: {
            list: async () => ({
                data: {
                    all: [
                        { id: "local", models: { qwen: { variants: { xhigh: {} } } } },
                        { id: "openai", models: { "gpt-6-luna": { variants: { high: {} } } } },
                    ],
                },
            }),
        },
        session: {
            create: async ({ body }: any) => {
                assert.deepEqual(body.model, { providerID: "openai", id: "gpt-6-luna" })
                return { data: { id: "luna-scratch" } }
            },
            prompt: async ({ body }: any) => {
                assert.deepEqual(body.model, { providerID: "openai", modelID: "gpt-6-luna" })
                assert.equal(body.variant, "high")
                return { data: { parts: [{ type: "text", text: validSummary }] } }
            },
            delete: async () => ({ data: true }),
        },
    }
    const summaries = await summarizeBoundaryJobs({
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        parentSessionId: "parent-session",
        jobs: [job],
        params: { ...params, variant: "xhigh" },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high",
    })
    assert.equal(summaries[job.key], validSummary)
})

test("inherit does not send the chat model variant to a different summary model", async () => {
    const logger = new Logger(false)
    const sdk = {
        session: {
            create: async () => ({ data: { id: "luna-scratch" } }),
            prompt: async ({ body }: any) => {
                assert.deepEqual(body.model, { providerID: "openai", modelID: "gpt-6-luna" })
                assert.equal(body.variant, undefined)
                return { data: { parts: [{ type: "text", text: validSummary }] } }
            },
            delete: async () => ({ data: true }),
        },
    }
    const summaries = await summarizeBoundaryJobs({
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        parentSessionId: "parent-session",
        jobs: [job],
        params,
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "inherit",
    })
    assert.equal(summaries[job.key], validSummary)
})

test("one scratch call returns separate validated summaries for multiple turns", async () => {
    const logger = new Logger(false)
    const jobs = [
        job,
        {
            ...job,
            key: "run-2",
            rangeStartMessageId: "msg-second",
            rangeEndMessageId: "msg-second",
            prompt: "Summarize a second independent turn.",
        },
        {
            ...job,
            key: "run-3",
            prompt: "Summarize a third independent turn.",
        },
        {
            ...job,
            key: "run-4",
            prompt: "Summarize a fourth independent turn.",
        },
    ]
    let calls = 0
    const sdk = {
        session: {
            create: async () => ({ data: { id: "batched-scratch" } }),
            prompt: async ({ body }: any) => {
                calls++
                assert.match(body.parts[0].text, /Job 1/)
                assert.match(body.parts[0].text, /Job 2/)
                assert.match(body.parts[0].text, /within about 8000 characters/)
                return {
                    data: {
                        parts: [
                            {
                                type: "text",
                                text: `\`\`\`json\n${JSON.stringify({
                                    [job.key]: validSummary,
                                    "run-2": validSummary.replace("src/app.ts", "src/second.ts"),
                                    "run-3": validSummary,
                                    "run-4": validSummary,
                                })}\n\`\`\``,
                            },
                        ],
                    },
                }
            },
            delete: async () => ({ data: true }),
        },
    }
    const summaries = await summarizeBoundaryJobs({
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        parentSessionId: "parent-batched",
        jobs,
        params,
        summaryEffort: "inherit",
    })
    assert.equal(calls, 1)
    assert.equal(summaries[job.key], validSummary)
    assert.match(summaries["run-2"], /src\/second.ts/)
    assert.equal(Object.keys(summaries).length, 4)
})

test("scratch create errors are reported instead of silently skipping jobs", async () => {
    const warnings: string[] = []
    const logger = new Logger(false)
    logger.warn = (message, data) => {
        warnings.push(`${message}: ${data?.error}`)
        return Promise.resolve()
    }
    const sdk = {
        session: {
            create: async () => ({ error: { message: "model.id required" } }),
            prompt: async () => {
                throw new Error("prompt should not run")
            },
            delete: async () => {
                throw new Error("delete should not run")
            },
        },
    }
    const summaries = await summarizeBoundaryJobs({
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        parentSessionId: "parent-session",
        jobs: [job],
        params,
        summaryEffort: "inherit",
    })

    assert.deepEqual(summaries, {})
    assert.ok(
        warnings.some((warning) =>
            warning.includes("Scratch session creation failed: model.id required"),
        ),
    )
})
