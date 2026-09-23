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
