import {
    type BoundarySummaryJob,
    type SummarizeProgressEvent,
    type Summarizer,
    type SummaryEffort,
} from "@better-compact/core"
import type { SessionCreateData } from "@opencode-ai/sdk/v2"
import type { Logger } from "../logger"
import type { RuntimeState } from "../state"

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
    summaryEffort?: SummaryEffort
    onProgress?: (event: SummarizeProgressEvent) => Promise<void> | void
}

export async function summarizeBoundaryJobs(
    input: SummarizeBoundaryJobsInput,
): Promise<Record<string, string>> {
    if (input.summaryEffort === "off") return {}
    if (input.jobs.length === 0 || !canRunScratchSession(input.client)) return {}
    const variant = await resolveCompactionVariant(input.client, input.params, input.summaryEffort)
    return input.runtime.summaryScheduler.summarize({
        sessionKey: input.parentSessionId,
        jobs: input.jobs,
        summarizer: createScratchSummarizer({ ...input, params: { ...input.params, variant } }),
        concurrency: input.concurrency,
        onProgress: input.onProgress,
    })
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

// Side-model transport: one throwaway OpenCode scratch session per job.
function createScratchSummarizer(input: SummarizeBoundaryJobsInput): Summarizer {
    return {
        async complete(job) {
            let scratchSessionId: string | undefined
            let untrackScratch: (() => void) | undefined
            try {
                // Session creation uses model.id; prompting below uses model.modelID.
                const body = {
                    parentID: input.parentSessionId,
                    title: `Better Compact summary ${job.rangeStartMessageId}`,
                    agent: input.params.agent,
                    model:
                        input.params.providerId && input.params.modelId
                            ? { providerID: input.params.providerId, id: input.params.modelId }
                            : undefined,
                    metadata: {
                        betterCompactScratch: true,
                        parentSessionId: input.parentSessionId,
                        rangeStartMessageId: job.rangeStartMessageId,
                        rangeEndMessageId: job.rangeEndMessageId,
                    },
                } satisfies NonNullable<SessionCreateData["body"]>
                const created = await input.client.session.create({
                    body,
                })
                if (created?.error)
                    throw scratchResponseError("Scratch session creation", created.error)
                scratchSessionId = created?.data?.id ?? created?.id
                if (!scratchSessionId)
                    throw new Error("Scratch session creation returned no session ID")
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
                        parts: [{ type: "text", text: job.prompt }],
                    },
                })
                if (response?.error)
                    throw scratchResponseError("Scratch session prompt", response.error)
                return extractAssistantText(response?.data ?? response)
            } catch (error) {
                input.logger.warn("Better Compact scratch summarization failed", {
                    rangeStartMessageId: job.rangeStartMessageId,
                    rangeEndMessageId: job.rangeEndMessageId,
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
        },
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
