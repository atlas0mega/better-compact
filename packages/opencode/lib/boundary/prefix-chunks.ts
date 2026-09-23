import {
    countTokens,
    formatSummarySections,
    prefixUserMessages,
    SUMMARY_SECTION_HEADERS,
    type BoundaryContextPlan,
    type BoundarySummaryJob,
    type Turn,
} from "@better-compact/core"
import { openCodeConventions } from "../codec"

const MAX_CHUNKS = 5
const PREFERRED_CHUNK_TOKENS = 23_000
const PROMPT_RESERVE_TOKENS = 200
const PROGRESS_MARKER = "- Resume from prior assistant progress: "

export interface PrefixChunk {
    job: BoundarySummaryJob
    count: number
}

// No sampling: if all old progress cannot fit in five bounded calls, retain
// the deterministic prefix instead of silently discarding a historical slice.
export function buildPrefixChunks(
    plan: BoundaryContextPlan,
    modelInputTokens = 24_000,
): PrefixChunk[] {
    const source = plan.prefixSummary
    if (!plan.requiresCustomCompaction || !source) return []
    const facts = source.split("\n").filter((line) => line.startsWith(PROGRESS_MARKER))
    if (facts.length === 0) return []
    const promptFor = (lines: string[]) =>
        [
            "Consolidate this chronological slice of historical assistant progress for the same ongoing task.",
            "Return the six Markdown headings in order: " + SUMMARY_SECTION_HEADERS.join("; "),
            "Preserve concrete decisions and WHY, changed paths/symbols, failures, constraints, completed and pending work, and exact errors. Distinguish old from current conclusions.",
            "Do not list every turn or invent facts. The original turns are available in the raw transcript for exact recall.",
            "Keep the entire structured summary under 4000 characters. Use '- (none)' for an empty section.",
            `Raw transcript: ${plan.transcript.relativePath}`,
            "",
            ...lines,
        ].join("\n")
    const factCosts = facts.map((fact) => countTokens(fact + "\n"))
    const total = factCosts.reduce((sum, cost) => sum + cost, 0)
    const available = modelInputTokens - countTokens(promptFor([])) - PROMPT_RESERVE_TOKENS
    if (available <= 0 || factCosts.some((cost) => cost > available)) return []
    const preferred = Math.min(PREFERRED_CHUNK_TOKENS, available)
    const initialCalls = Math.min(
        MAX_CHUNKS,
        facts.length,
        Math.max(1, Math.ceil(total / preferred)),
    )
    // Increase call count only if the balanced partition exceeds the model's
    // context. Never silently sample a slice to force a fit.
    for (let calls = initialCalls; calls <= Math.min(MAX_CHUNKS, facts.length); calls++) {
        const chunks = balancedChunks(facts, factCosts, calls)
        let offset = 0
        const planned = chunks.map((lines, index) => {
            const start = offset + 1
            offset += lines.length
            const prompt = `${promptFor(lines)}\n\nProgress entries ${start}-${offset} of ${facts.length}.`
            if (countTokens(prompt) + 80 > modelInputTokens) return null
            return {
                count: lines.length,
                job: {
                    key: `prefix-chunk:${plan.rangeHash}:${index}`,
                    rangeStartMessageId: plan.transcript.messageIds[0] ?? "unknown",
                    rangeEndMessageId: plan.transcript.messageIds.at(-1) ?? "unknown",
                    transcriptRelativePath: plan.transcript.relativePath,
                    prompt,
                },
            }
        })
        if (planned.every((chunk) => chunk !== null)) return planned as PrefixChunk[]
    }
    return []
}

function balancedChunks(facts: string[], costs: number[], calls: number): string[][] {
    const chunks: string[][] = []
    let cursor = 0
    let remaining = costs.reduce((sum, cost) => sum + cost, 0)
    for (let call = 0; call < calls; call++) {
        const slots = calls - call
        const target = remaining / slots
        const lines: string[] = []
        let used = 0
        while (cursor < facts.length - slots + 1) {
            const next = costs[cursor]
            if (lines.length > 0 && Math.abs(used - target) <= Math.abs(used + next - target)) break
            lines.push(facts[cursor++])
            used += next
        }
        chunks.push(lines)
        remaining -= used
    }
    return chunks
}

export function assemblePrefixChunks(
    chunks: PrefixChunk[],
    summaries: Record<string, string>,
    turns: Turn[],
    rawTailStartIndex: number,
): string | null {
    if (chunks.length === 0 || chunks.some((chunk) => !summaries[chunk.job.key])) return null
    const sections: string[][] = SUMMARY_SECTION_HEADERS.map(() => [])
    const rawTail = turns.slice(rawTailStartIndex)
    const users = prefixUserMessages(
        turns.slice(0, rawTailStartIndex),
        openCodeConventions,
        rawTail,
    )
    sections[4].push(...users)
    for (let index = 0; index < chunks.length; index++) {
        const summary = summaries[chunks[index].job.key]
        const lines = summary.split(/\r\n|\n|\r/)
        const offsets = SUMMARY_SECTION_HEADERS.map((header) =>
            lines.findIndex((line) => line.trim() === header),
        )
        if (offsets.some((offset) => offset < 0)) return null
        for (let section = 0; section < offsets.length; section++) {
            const body = lines
                .slice(offsets[section] + 1, offsets[section + 1] ?? lines.length)
                .join("\n")
                .trim()
            if (!body || body === "- (none)") continue
            sections[section].push(
                `[older segment ${index + 1}/${chunks.length}] ${body.replace(/^[-*]\s*/, "")}`,
            )
        }
    }
    return formatSummarySections(sections)
}
