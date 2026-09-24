import assert from "node:assert/strict"
import test from "node:test"
import { createSummaryScheduler, type BoundarySummaryJob, type Logger } from "@better-compact/core"

const job: BoundarySummaryJob = {
    key: "assistant-run",
    rangeStartMessageId: "msg-assistant-1",
    rangeEndMessageId: "msg-assistant-2",
    transcriptRelativePath: "transcripts/session/range.md",
    prompt: "Summarize the run.",
}

const summaryHeaders = [
    "## Decisions",
    "## Files & Symbols",
    "## Errors (verbatim)",
    "## What failed and why",
    "## Constraints",
    "## Next step",
]

const validSummary = [
    summaryHeaders[0],
    "- Keep the canonical parser.",
    "",
    summaryHeaders[1],
    "- src/parser.ts:parseInput",
    "",
    summaryHeaders[2],
    "- EINVAL request_id=req_123",
    "",
    summaryHeaders[3],
    "- The legacy parser rejected the payload.",
    "",
    summaryHeaders[4],
    "- Preserve call_456 exactly.",
    "",
    summaryHeaders[5],
    "- Run pnpm test.",
].join("\n")

function logger(warnings: Array<{ message: string; data: unknown }>): Logger {
    return {
        info() {},
        debug() {},
        warn(message, data) {
            warnings.push({ message, data })
        },
        error() {},
    }
}

test("summary scheduler rejects prose without the required sections", async () => {
    const warnings: Array<{ message: string; data: unknown }> = []
    const scheduler = createSummaryScheduler(logger(warnings))
    const summaries = await scheduler.summarize({
        sessionKey: "session-invalid",
        jobs: [job],
        summarizer: {
            complete: async () =>
                "Investigated the modules, ran the tests, and confirmed the implementation is ready for replay with all important details preserved.",
        },
    })

    assert.deepEqual(summaries, {})
    assert.equal(warnings.length, 1)
})

test("summary scheduler accepts a structured failure-preserving summary", async () => {
    const scheduler = createSummaryScheduler(logger([]))
    const summaries = await scheduler.summarize({
        sessionKey: "session-valid",
        jobs: [job],
        summarizer: { complete: async () => `\n${validSummary}\n` },
    })

    assert.equal(summaries[job.key], validSummary)
})

test("summary scheduler accepts an oversized structured summary", async () => {
    const overlongSummary = validSummary.replace(
        "- Keep the canonical parser.",
        `- ${"x".repeat(4_000)}`,
    )
    assert.ok(overlongSummary.length > 4_000)

    const scheduler = createSummaryScheduler(logger([]))
    const summaries = await scheduler.summarize({
        sessionKey: "session-overlong",
        jobs: [job],
        summarizer: { complete: async () => overlongSummary },
    })

    const stored = summaries[job.key]
    assert.ok(stored)
    assert.ok(stored.length <= 4_000)
    let previousHeaderIndex = -1
    const storedLines = stored.split(/\r\n|\n|\r/).map((line) => line.trim())
    for (const header of summaryHeaders) {
        const headerIndex = storedLines.indexOf(header, previousHeaderIndex + 1)
        assert.ok(headerIndex > previousHeaderIndex)
        previousHeaderIndex = headerIndex
    }
    assert.match(stored, /^- Run pnpm test\.$/m)
})

test("chunk summaries reject oversized output rather than silently truncating facts", async () => {
    const warnings: Array<{ message: string; data: unknown }> = []
    const summaries = await createSummaryScheduler(logger(warnings)).summarize({
        sessionKey: "session-prefix-chunk",
        jobs: [job],
        summarizer: {
            complete: async () =>
                validSummary.replace("- Keep the canonical parser.", `- ${"x".repeat(4_000)}`),
        },
        rejectOversized: true,
    })
    assert.deepEqual(summaries, {})
    assert.ok(warnings.some((warning) => warning.message.includes("overlong")))
})

test("a prefix chunk may return more than a single-turn 4000-character limit", async () => {
    const expanded = validSummary.replace("- Keep the canonical parser.", `- ${"specific decision ".repeat(500)}`)
    assert.ok(expanded.length > 4_000)
    const summaries = await createSummaryScheduler(logger([])).summarize({
        sessionKey: "session-large-prefix-chunk",
        jobs: [job],
        summarizer: {
            complete: async () => expanded,
            completeBatch: async () => ({ [job.key]: expanded }),
        },
        maxCalls: 1,
        rejectOversized: true,
        maxSummaryChars: 16_000,
    })
    assert.equal(summaries[job.key], expanded)
})

test("summary scheduler rejects a too-short response", async () => {
    const warnings: Array<{ message: string; data: unknown }> = []
    const scheduler = createSummaryScheduler(logger(warnings))
    const summaries = await scheduler.summarize({
        sessionKey: "session-too-short",
        jobs: [job],
        summarizer: { complete: async () => "Too short." },
    })

    assert.deepEqual(summaries, {})
    assert.equal(warnings.length, 1)
})

test("a thrown summarizer is contained and reported as a failed job", async () => {
    const warnings: Array<{ message: string; data: unknown }> = []
    const progress: boolean[] = []
    const scheduler = createSummaryScheduler(logger(warnings))

    const summaries = await scheduler.summarize({
        sessionKey: "session-throw",
        jobs: [job],
        summarizer: {
            complete: async () => {
                throw new Error("provider unavailable")
            },
        },
        onProgress: (event) => {
            progress.push(event.ok)
        },
    })

    assert.deepEqual(summaries, {})
    assert.deepEqual(progress, [false])
    assert.ok(warnings.some((entry) => entry.message === "Summary job failed"))
})

test("three consecutive failures open the session breaker until its window elapses", async () => {
    let now = 1_000
    let calls = 0
    const warnings: Array<{ message: string; data: unknown }> = []
    const scheduler = createSummaryScheduler(logger(warnings), { now: () => now })
    const failing = {
        complete: async () => {
            calls++
            return null
        },
    }

    for (let attempt = 0; attempt < 3; attempt++) {
        assert.deepEqual(
            await scheduler.summarize({
                sessionKey: "session-breaker",
                jobs: [job],
                summarizer: failing,
            }),
            {},
        )
    }
    await scheduler.summarize({
        sessionKey: "session-breaker",
        jobs: [job],
        summarizer: failing,
    })
    assert.equal(calls, 3)
    assert.ok(warnings.some((entry) => entry.message === "Summary circuit breaker opened"))

    const otherSession = await scheduler.summarize({
        sessionKey: "session-healthy",
        jobs: [job],
        summarizer: { complete: async () => validSummary },
    })
    assert.equal(otherSession[job.key], validSummary)

    now += 5 * 60_000
    const recovered = await scheduler.summarize({
        sessionKey: "session-breaker",
        jobs: [job],
        summarizer: {
            complete: async () => {
                calls++
                return validSummary
            },
        },
    })
    assert.equal(calls, 4)
    assert.equal(recovered[job.key], validSummary)
})

test("the breaker stops scheduling a failing batch after the threshold", async () => {
    let calls = 0
    const scheduler = createSummaryScheduler(logger([]))
    const jobs = Array.from({ length: 6 }, (_, index) => ({
        ...job,
        key: `assistant-run-${index}`,
        rangeStartMessageId: `msg-assistant-${index * 2}`,
        rangeEndMessageId: `msg-assistant-${index * 2 + 1}`,
    }))

    const summaries = await scheduler.summarize({
        sessionKey: "session-batch",
        jobs,
        concurrency: 1,
        summarizer: {
            complete: async () => {
                calls++
                return null
            },
        },
    })

    assert.deepEqual(summaries, {})
    assert.equal(calls, 3)
})

test("an interleaved success resets the consecutive failure count", async () => {
    let calls = 0
    const outcomes: Array<string | null> = [null, null, validSummary, null, null, null]
    const scheduler = createSummaryScheduler(logger([]))
    const summarizer = {
        complete: async () => {
            calls++
            return outcomes.shift() ?? null
        },
    }

    for (let attempt = 0; attempt < 6; attempt++) {
        await scheduler.summarize({
            sessionKey: "session-reset",
            jobs: [job],
            summarizer,
        })
    }
    await scheduler.summarize({
        sessionKey: "session-reset",
        jobs: [job],
        summarizer,
    })

    assert.equal(calls, 6)
})

test("grouped summaries scale to five balanced calls per compaction", async () => {
    const batches: BoundarySummaryJob[][] = []
    let active = 0
    let maxActive = 0
    const scheduler = createSummaryScheduler(logger([]))
    const jobs = Array.from({ length: 40 }, (_, index) => ({
        ...job,
        key: `run-${index}`,
        prompt: `Summarize turn ${index}: ${"x".repeat(4_000)}`,
    }))
    const request = {
        sessionKey: "session-cost-budget",
        jobs,
        maxCalls: 5,
        concurrency: 5,
        targetBatchTokens: 5_000,
        maxBatchTokens: 10_000,
        maxJobsPerBatch: 16,
        summarizer: {
            complete: async () => {
                throw new Error("single-turn transport must not run")
            },
            completeBatch: async (group: BoundarySummaryJob[]) => {
                active++
                maxActive = Math.max(maxActive, active)
                await new Promise((resolve) => setTimeout(resolve, 1))
                batches.push(group)
                active--
                return Object.fromEntries(group.map((item) => [item.key, validSummary]))
            },
        },
    }

    const first = await scheduler.summarize(request)
    assert.equal(Object.keys(first).length, 40)
    assert.equal(batches.length, 5)
    assert.equal(maxActive, 5)
    assert.deepEqual(
        batches.map((group) => group.length),
        [8, 8, 8, 8, 8],
    )
    assert.equal(Object.keys(await scheduler.summarize(request)).length, 40)
    assert.equal(batches.length, 10)
})

test("grouped summaries sample evenly across history when the input budget is exceeded", async () => {
    const groups: BoundarySummaryJob[][] = []
    const scheduler = createSummaryScheduler(logger([]))
    const jobs = Array.from({ length: 100 }, (_, index) => ({
        ...job,
        key: `run-${index}`,
        prompt: `Historical turn ${index}: ${"x".repeat(4_000)}`,
    }))
    const summaries = await scheduler.summarize({
        sessionKey: "session-spread",
        jobs,
        maxCalls: 5,
        targetBatchTokens: 5_000,
        maxBatchTokens: 6_000,
        maxJobsPerBatch: 10,
        summarizer: {
            complete: async () => null,
            completeBatch: async (group) => {
                groups.push(group)
                return Object.fromEntries(group.map((item) => [item.key, validSummary]))
            },
        },
    })
    const indices = Object.keys(summaries).map((key) => Number(key.slice(4)))
    assert.equal(groups.length, 5)
    assert.ok(indices.length > 0 && indices.length < 100)
    assert.ok(Math.min(...indices) < 5)
    assert.ok(Math.max(...indices) > 95)
    assert.ok(groups.every((group) => group.length <= 10))
})

test("many short turns use one call when their combined input fits the target", async () => {
    const groups: BoundarySummaryJob[][] = []
    const scheduler = createSummaryScheduler(logger([]))
    const jobs = Array.from({ length: 100 }, (_, index) => ({ ...job, key: `run-${index}` }))
    const summaries = await scheduler.summarize({
        sessionKey: "session-short-turns",
        jobs,
        maxCalls: 5,
        targetBatchTokens: 12_000,
        maxJobsPerBatch: 16,
        summarizer: {
            complete: async () => null,
            completeBatch: async (group) => {
                groups.push(group)
                return Object.fromEntries(group.map((item) => [item.key, validSummary]))
            },
        },
    })
    assert.equal(groups.length, 1)
    assert.equal(Object.keys(summaries).length, 16)
    assert.ok(groups[0].some((item) => item.key === "run-96"))
})

test("grouped summary failures stop after three model calls", async () => {
    let calls = 0
    const scheduler = createSummaryScheduler(logger([]))
    const jobs = Array.from({ length: 10 }, (_, index) => ({ ...job, key: `run-${index}` }))
    await scheduler.summarize({
        sessionKey: "session-group-breaker",
        jobs,
        concurrency: 1,
        maxCalls: 5,
        targetBatchTokens: 1,
        maxJobsPerBatch: 2,
        summarizer: {
            complete: async () => null,
            completeBatch: async () => {
                calls++
                return null
            },
        },
    })
    assert.equal(calls, 3)
})
