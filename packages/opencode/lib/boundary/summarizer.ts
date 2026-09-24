import {
    countTokens,
    type BoundaryContextPlan,
    type BoundarySummaryJob,
    type SummarizeProgressEvent,
    type Summarizer,
    type SummaryEffort,
    type Turn,
} from "@better-compact/core"
import type { SessionCreateData } from "@opencode-ai/sdk/v2"
import type { Logger } from "../logger"
import type { RuntimeState } from "../state"
import { assemblePrefixChunks, buildPrefixChunks } from "./prefix-chunks"

interface SummarizeBoundaryJobsInput {
    client: any
    runtime: RuntimeState
    logger: Logger
    parentSessionId: string
    jobs: BoundarySummaryJob[]
    params: {
        providerId: string | undefined
        modelId: string | undefined
        agent: string | undefined
        variant: string | undefined
    }
    concurrency?: number
    maxBatchTokens?: number
    maxJobsPerBatch?: number
    rejectOversized?: boolean
    summaryEffort?: SummaryEffort
    summaryModel?: string | null
    onProgress?: (event: SummarizeProgressEvent) => Promise<void> | void
}

export async function summarizeBoundaryJobs(
    input: SummarizeBoundaryJobsInput,
): Promise<Record<string, string>> {
    if (input.summaryEffort === "off") return {}
    if (input.jobs.length === 0 || !canRunScratchSession(input.client)) return {}
    const params = summaryModelParams(input)
    const variant = await resolveCompactionVariant(input.client, params, input.summaryEffort)
    return input.runtime.summaryScheduler.summarize({
        sessionKey: input.parentSessionId,
        jobs: input.jobs,
        summarizer: createScratchSummarizer({ ...input, params: { ...params, variant } }),
        concurrency: input.concurrency,
        maxCalls: 5,
        maxBatchTokens: input.maxBatchTokens,
        maxJobsPerBatch: input.maxJobsPerBatch,
        rejectOversized: input.rejectOversized,
        onProgress: input.onProgress,
    })
}

export async function summarizePrefixChunks(
    input: Omit<SummarizeBoundaryJobsInput, "jobs"> & {
        plan: BoundaryContextPlan
        turns: Turn[]
    },
): Promise<string | null | undefined> {
    if (input.summaryEffort === "off" || !canRunScratchSession(input.client)) return undefined
    const target = summaryModelParams(input)
    const context =
        target.providerId && target.modelId
            ? await input.runtime.resolveModelLimit(target.providerId, target.modelId)
            : undefined
    if (!context) {
        input.logger.warn("Cannot size prefix chunks without the summary model context limit", {
            sessionId: input.parentSessionId,
            providerId: target.providerId,
            modelId: target.modelId,
        })
        return undefined
    }
    // Leave headroom for OpenCode's instructions, the JSON wrapper and output.
    const modelInputTokens = context - Math.max(8_192, Math.ceil(context * 0.15))
    const chunks = buildPrefixChunks(input.plan, modelInputTokens)
    if (chunks.length === 0) return undefined
    const summaries = await summarizeBoundaryJobs({
        ...input,
        jobs: chunks.map((chunk) => chunk.job),
        maxBatchTokens: modelInputTokens,
        maxJobsPerBatch: 1,
        rejectOversized: true,
    })
    if (chunks.some((chunk) => !summaries[chunk.job.key])) return null
    if (
        chunks.some(
            (chunk) => countTokens(summaries[chunk.job.key]) < Math.min(400, chunk.count * 2),
        )
    ) {
        input.logger.warn("Chunked prefix summary omitted too much historical progress", {
            sessionId: input.parentSessionId,
        })
        return null
    }
    const assembled = assemblePrefixChunks(
        chunks,
        summaries,
        input.turns,
        input.plan.rawTailStartIndex,
    )
    return assembled && countTokens(assembled) < countTokens(input.plan.prefixSummary ?? "")
        ? assembled
        : null
}

function summaryModelParams(
    input: Pick<SummarizeBoundaryJobsInput, "params" | "summaryModel">,
): SummarizeBoundaryJobsInput["params"] {
    const model = input.summaryModel
    const separator = model?.indexOf("/") ?? -1
    if (!model || separator <= 0 || separator >= model.length - 1) return input.params
    const providerId = model.slice(0, separator)
    const modelId = model.slice(separator + 1)
    return {
        ...input.params,
        providerId,
        modelId,
        // A variant from the conversation model may not exist on this model.
        variant:
            providerId === input.params.providerId && modelId === input.params.modelId
                ? input.params.variant
                : undefined,
    }
}

/** Resolve only advertised variants; unsupported efforts retain the active variant. */
export async function resolveCompactionVariant(
    client: any,
    params: SummarizeBoundaryJobsInput["params"],
    effort?: SummaryEffort,
): Promise<string | undefined> {
    if (!effort || effort === "inherit" || effort === "off") return params.variant
    try {
        const response = await client.provider.list()
        const payload = response?.data ?? response
        const providers = Array.isArray(payload)
            ? payload
            : (payload?.all ?? payload?.providers ?? [])
        const provider = providers.find((item: any) => item?.id === params.providerId)
        const variants = provider?.models?.[params.modelId ?? ""]?.variants ?? {}
        const candidates = effort === "max" ? ["max", "xhigh"] : [effort]
        return candidates.find((candidate) => Object.hasOwn(variants, candidate)) ?? params.variant
    } catch {
        return params.variant
    }
}

function canRunScratchSession(client: any): boolean {
    return (
        typeof client?.session?.create === "function" &&
        typeof client?.session?.prompt === "function" &&
        typeof client?.session?.delete === "function"
    )
}

// Side-model transport: one throwaway OpenCode scratch session per call (one or more jobs).
function createScratchSummarizer(input: SummarizeBoundaryJobsInput): Summarizer {
    return {
        complete: (job) => runScratchSummary(input, [job], job.prompt),
        async completeBatch(jobs) {
            const prefixChunks = jobs.every((job) => job.key.startsWith("prefix-chunk:"))
            const prompt = [
                prefixChunks
                    ? `Consolidate ${jobs.length} chronological history segments.`
                    : `Summarize ${jobs.length} independent historical assistant turns.`,
                "Return ONLY a JSON object mapping each job key to its own Markdown summary.",
                "Each summary must contain the six headings specified in its job prompt, in order.",
                `Keep each summary within 4000 characters and the full JSON response within about ${Math.ceil(4_000 * Math.sqrt(jobs.length))} characters. Preserve each assigned ${prefixChunks ? "segment" : "turn"}; do not combine jobs or omit keys.`,
                ...jobs.map((job, index) =>
                    [`\n--- Job ${index + 1}: ${JSON.stringify(job.key)} ---`, job.prompt].join(
                        "\n",
                    ),
                ),
            ].join("\n")
            const text = await runScratchSummary(input, jobs, prompt)
            if (!text) return null
            const unwrapped = text
                .trim()
                .replace(/^```(?:markdown|md|json)?\s*\n?/i, "")
                .replace(/\n?```$/, "")
                .trim()
            if (jobs.length === 1 && unwrapped.startsWith("## Decisions")) {
                return { [jobs[0].key]: unwrapped }
            }
            try {
                const json = unwrapped
                    .trim()
                    .replace(/^```(?:json)?\s*\n?/i, "")
                    .replace(/\n?```$/, "")
                const parsed: unknown = JSON.parse(json)
                if (!parsed || typeof parsed !== "object" || Array.isArray(parsed))
                    throw new Error("Expected a JSON object")
                const record = parsed as Record<string, unknown>
                return Object.fromEntries(
                    jobs
                        .filter((job) => typeof record[job.key] === "string")
                        .map((job) => [job.key, record[job.key] as string]),
                )
            } catch (error) {
                input.logger.warn("Invalid grouped scratch summary response", {
                    jobs: jobs.length,
                    error: error instanceof Error ? error.message : String(error),
                })
                return null
            }
        },
    }
}

async function runScratchSummary(
    input: SummarizeBoundaryJobsInput,
    jobs: BoundarySummaryJob[],
    prompt: string,
): Promise<string | null> {
    const first = jobs[0]
    const last = jobs.at(-1) ?? first
    let scratchSessionId: string | undefined
    let untrackScratch: (() => void) | undefined
    try {
        // Session creation uses model.id; prompting below uses model.modelID.
        const body = {
            parentID: input.parentSessionId,
            title: `Better Compact summary ${first.rangeStartMessageId}`,
            agent: input.params.agent,
            model:
                input.params.providerId && input.params.modelId
                    ? { providerID: input.params.providerId, id: input.params.modelId }
                    : undefined,
            metadata: {
                betterCompactScratch: true,
                parentSessionId: input.parentSessionId,
                rangeStartMessageId: first.rangeStartMessageId,
                rangeEndMessageId: last.rangeEndMessageId,
            },
        } satisfies NonNullable<SessionCreateData["body"]>
        const created = await input.client.session.create({
            body,
        })
        if (created?.error) throw scratchResponseError("Scratch session creation", created.error)
        scratchSessionId = created?.data?.id ?? created?.id
        if (!scratchSessionId) throw new Error("Scratch session creation returned no session ID")
        untrackScratch = input.runtime.trackScratch(scratchSessionId)

        const response = await input.client.session.prompt({
            path: { id: scratchSessionId },
            body: {
                agent: input.params.agent,
                model:
                    input.params.providerId && input.params.modelId
                        ? {
                              providerID: input.params.providerId,
                              modelID: input.params.modelId,
                          }
                        : undefined,
                variant: input.params.variant,
                parts: [{ type: "text", text: prompt }],
            },
        })
        if (response?.error) throw scratchResponseError("Scratch session prompt", response.error)
        return extractAssistantText(response?.data ?? response)
    } catch (error) {
        input.logger.warn("Better Compact scratch summarization failed", {
            rangeStartMessageId: first.rangeStartMessageId,
            rangeEndMessageId: last.rangeEndMessageId,
            error: error instanceof Error ? error.message : String(error),
        })
        return null
    } finally {
        if (scratchSessionId) {
            try {
                await input.client.session.delete({ path: { id: scratchSessionId } })
            } catch (error) {
                input.logger.warn("Failed to delete Better Compact scratch session", {
                    scratchSessionId,
                    error: error instanceof Error ? error.message : String(error),
                })
            } finally {
                untrackScratch?.()
            }
        }
    }
}

function scratchResponseError(operation: string, error: any): Error {
    const message = error instanceof Error ? error.message : error?.message
    return new Error(
        `${operation} failed: ${typeof message === "string" ? message : "server returned an error"}`,
    )
}

function extractAssistantText(message: any): string {
    const parts = Array.isArray(message?.parts) ? message.parts : []
    return parts
        .filter((part: any) => part?.type === "text" && typeof part.text === "string")
        .map((part: any) => part.text)
        .join("\n\n")
        .trim()
}
