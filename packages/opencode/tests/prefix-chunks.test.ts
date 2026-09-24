import assert from "node:assert/strict"
import test from "node:test"
import {
    countTokens,
    formatPrefixSummary,
    type BoundaryContextPlan,
    type Turn,
} from "@better-compact/core"
import { assemblePrefixChunks, buildPrefixChunks } from "../lib/boundary/prefix-chunks"
import { summarizePrefixChunks } from "../lib/boundary/summarizer"
import { createRuntimeState } from "../lib/state"
import { Logger } from "../lib/logger"

const userText = "Keep the original requirement.\n## Next step\nVerify the final build."

function history(count: number): { plan: BoundaryContextPlan; turns: Turn[] } {
    const turns: Turn[] = [
        {
            key: "user-original",
            stamp: 1,
            role: "user",
            items: [{ kind: "synthetic", key: "user-text", text: userText }],
        },
        ...Array.from({ length: count }, (_, index): Turn => ({
            key: `assistant-${index}`,
            stamp: index + 2,
            role: "assistant",
            items: [
                {
                    kind: "synthetic",
                    key: `text-${index}`,
                    text: `Worked on src/feature-${index}.ts: ${"concrete implementation detail ".repeat(17)}`,
                },
            ],
        })),
        {
            key: "user-current",
            stamp: count + 3,
            role: "user",
            items: [{ kind: "synthetic", key: "current-text", text: "Continue" }],
        },
    ]
    return {
        turns,
        plan: {
            requiresCustomCompaction: true,
            rangeHash: "stable-range",
            rawTailStartIndex: turns.length - 1,
            prefixSummary: formatPrefixSummary(turns.slice(0, -1)),
            transcript: {
                relativePath: "transcripts/raw.md",
                content: "",
                messageIds: turns.slice(0, -1).map((turn) => turn.key),
            },
        } as BoundaryContextPlan,
    }
}

function checkpoint(index: number): string {
    return [
        "## Decisions",
        `- Resolved segment ${index} after reviewing its work.`,
        "## Files & Symbols",
        `- src/feature-${index}.ts`,
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- (none)",
        "## Constraints",
        `- Keep files from segment ${index} consistent.`,
        "## Next step",
        `- Check progress from segment ${index}.`,
    ].join("\n")
}

test("a long deterministic prefix is covered by five bounded chronological jobs", () => {
    const { plan, turns } = history(720)
    const jobs = buildPrefixChunks(plan)
    assert.equal(jobs.length, 5)
    assert.equal(
        jobs.reduce((sum, chunk) => sum + chunk.count, 0),
        720,
    )
    for (const chunk of jobs) {
        assert.ok(countTokens(chunk.job.prompt) + 80 <= 24_000)
        assert.match(chunk.job.prompt, /transcripts\/raw\.md/)
    }
    const sizes = jobs.map((chunk) => countTokens(chunk.job.prompt))
    assert.ok(Math.max(...sizes) - Math.min(...sizes) < 500, `unbalanced prompts: ${sizes}`)
    assert.match(jobs[0].job.prompt, /feature-0\.ts/)
    assert.match(jobs.at(-1)!.job.prompt, /feature-719\.ts/)
    const results = Object.fromEntries(jobs.map((job, index) => [job.job.key, checkpoint(index)]))
    const assembled = assemblePrefixChunks(jobs, results, turns, plan.rawTailStartIndex)
    assert.ok(assembled)
    assert.ok(assembled.includes(`- ${userText}`))
    assert.ok(assembled.indexOf("Resolved segment 0") < assembled.indexOf("Resolved segment 4"))
    assert.ok(countTokens(assembled) < countTokens(plan.prefixSummary!))
    assert.equal(
        assemblePrefixChunks(
            jobs,
            { ...results, [jobs[3].job.key]: "" },
            turns,
            plan.rawTailStartIndex,
        ),
        null,
    )
})

test("the chunk planner refuses oversized histories rather than omitting older turns", () => {
    const { plan } = history(1_400)
    assert.deepEqual(buildPrefixChunks(plan), [])
    const largerModel = buildPrefixChunks(plan, 100_000)
    assert.equal(largerModel.length, 5)
    assert.equal(
        largerModel.reduce((sum, chunk) => sum + chunk.count, 0),
        1_400,
    )
    const sizes = largerModel.map((chunk) => countTokens(chunk.job.prompt))
    assert.ok(Math.max(...sizes) - Math.min(...sizes) < 500, `unbalanced prompts: ${sizes}`)
    assert.ok(sizes.every((size) => size + 80 <= 100_000))
    assert.deepEqual(buildPrefixChunks(history(720).plan, 20_000), [])
    const smallHistory = buildPrefixChunks(history(40).plan, 20_000)
    assert.equal(smallHistory.length, 1)
    assert.equal(smallHistory[0].count, 40)
})

test("different-sized progress entries balance by tokens while staying chronological", () => {
    const { plan, turns } = history(720)
    for (let index = 0; index < 720; index += 4) {
        turns[index + 1].items = [
            { kind: "synthetic", key: `text-${index}`, text: `Finished src/feature-${index}.ts.` },
        ]
    }
    plan.prefixSummary = formatPrefixSummary(turns.slice(0, -1))
    const chunks = buildPrefixChunks(plan)
    assert.ok(chunks.length > 1 && chunks.length <= 5)
    assert.equal(
        chunks.reduce((sum, chunk) => sum + chunk.count, 0),
        720,
    )
    const sizes = chunks.map((chunk) => countTokens(chunk.job.prompt))
    assert.ok(Math.max(...sizes) - Math.min(...sizes) < 500, `unbalanced prompts: ${sizes}`)
    assert.match(chunks[0].job.prompt, /feature-0\.ts/)
    assert.match(chunks.at(-1)!.job.prompt, /feature-719\.ts/)
})

test("five independent Luna/high chunks complete concurrently and assemble in source order", async () => {
    const { plan, turns } = history(704)
    let calls = 0
    let active = 0
    let peak = 0
    let failChunk = -1
    const sdk = {
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "openai",
                            models: {
                                "gpt-6-luna": {
                                    variants: { high: {} },
                                    limit: { context: 262_144 },
                                },
                            },
                        },
                    ],
                },
            }),
        },
        session: {
            create: async ({ body }: any) => {
                assert.equal(body.parentID, "parent")
                assert.equal(body.agent, "tennis-mc-specialist")
                return { data: { id: `chunk-scratch-${++calls}` } }
            },
            prompt: async ({ body }: any) => {
                assert.deepEqual(body.model, { providerID: "openai", modelID: "gpt-6-luna" })
                assert.equal(body.variant, "high")
                assert.equal(body.agent, "tennis-mc-specialist")
                active++
                peak = Math.max(peak, active)
                const index = Number(
                    body.parts[0].text.match(/Job 1: "prefix-chunk:stable-range:(\d)"/)?.[1] ?? -1,
                )
                assert.ok(index >= 0)
                await new Promise((resolve) => setTimeout(resolve, 5))
                active--
                if (index === failChunk)
                    return { data: { parts: [{ type: "text", text: "Incomplete checkpoint" }] } }
                const detail = Array.from(
                    { length: 25 },
                    (_, item) =>
                        `- Decision ${index}.${item} from src/feature-${index}.ts retains context.`,
                ).join("\n")
                const summary = checkpoint(index).replace(
                    `- Resolved segment ${index} after reviewing its work.`,
                    detail,
                )
                return {
                    data: {
                        parts: [
                            {
                                type: "text",
                                text: index === 2 ? `\`\`\`markdown\n${summary}\n\`\`\`` : summary,
                            },
                        ],
                    },
                }
            },
            delete: async () => ({ data: true }),
        },
    }
    const logger = new Logger(false)
    const input = {
        client: sdk,
        runtime: createRuntimeState(sdk, logger),
        logger,
        parentSessionId: "parent",
        plan,
        turns,
        params: {
            providerId: "vast",
            modelId: "qwen",
            agent: "tennis-mc-specialist",
            variant: "low",
        },
        summaryModel: "openai/gpt-6-luna",
        summaryEffort: "high" as const,
        concurrency: 5,
    }
    const result = await summarizePrefixChunks(input)
    assert.equal(calls, 5)
    assert.equal(peak, 5)
    assert.ok(result?.includes(`- ${userText}`))
    assert.ok(result!.indexOf("Decision 0.0") < result!.indexOf("Decision 4.0"))
    assert.ok(countTokens(result!) < countTokens(plan.prefixSummary!))
    failChunk = 2
    assert.equal(await summarizePrefixChunks(input), null)
    assert.equal(calls, 10)
})

test("prefix synthesis sizes balanced calls against the chosen summary model", async () => {
    const { plan, turns } = history(1_400)
    for (const context of [30_000, 262_144]) {
        let calls = 0
        const sdk = {
            provider: {
                list: async () => ({
                    data: {
                        all: [
                            {
                                id: "openai",
                                models: { "gpt-6-luna": { limit: { context } } },
                            },
                        ],
                    },
                }),
            },
            session: {
                create: async () => ({ data: { id: `scratch-${++calls}` } }),
                prompt: async ({ body }: any) => {
                    const prompt = body.parts[0].text as string
                    const index = Number(
                        prompt.match(/Job 1: "prefix-chunk:stable-range:(\d)"/)?.[1],
                    )
                    assert.ok(index >= 0 && index < 5)
                    assert.ok(
                        countTokens(prompt) < context - Math.max(8_192, Math.ceil(context * 0.15)),
                    )
                    return {
                        data: {
                            parts: [
                                {
                                    type: "text",
                                    text: checkpoint(index).replace(
                                        `- Resolved segment ${index} after reviewing its work.`,
                                        Array.from(
                                            { length: 70 },
                                            (_, item) => `- Resolved decision ${index}.${item}.`,
                                        ).join("\n"),
                                    ),
                                },
                            ],
                        },
                    }
                },
                delete: async () => ({ data: true }),
            },
        }
        const logger = new Logger(false)
        const result = await summarizePrefixChunks({
            client: sdk,
            runtime: createRuntimeState(sdk, logger),
            logger,
            parentSessionId: "large-history",
            plan,
            turns,
            params: { providerId: "vast", modelId: "qwen", agent: "assistant", variant: "low" },
            summaryModel: "openai/gpt-6-luna",
            summaryEffort: "inherit",
            concurrency: 5,
        })
        if (context === 30_000) {
            assert.equal(result, undefined)
            assert.equal(calls, 0)
        } else {
            assert.equal(calls, 5)
            assert.ok(result?.includes(`- ${userText}`))
            assert.ok(
                result!.indexOf("Resolved decision 0.0") < result!.indexOf("Resolved decision 4.0"),
            )
        }
    }
})
