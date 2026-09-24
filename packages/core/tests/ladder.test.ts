import assert from "node:assert/strict"
import test from "node:test"
import {
    assistantRunsStage,
    buildPlan,
    countTokens,
    createEngine,
    createSummaryScheduler,
    reasoningStage,
    replayPlanSnapshot,
    purgeErrorInputsStage,
    skillsStage,
    supersedeReadsStage,
    toolsOldStage,
    toolsRemainingStage,
    toPlanSnapshot,
    transformTurns,
    type BuildPlanInputs,
    type CodecOps,
    type Conventions,
    type Item,
    type LadderSpec,
    type PlanSnapshot,
    type Turn,
} from "@better-compact/core"

// A minimal platform: handles carry simple part records and the codec prices
// turns as the JSON model shape a host would serialize, mirroring how real
// codecs price their own native forms.

interface TestTextPart {
    type: "text"
    text: string
}

interface TestReasoningPart {
    type: "reasoning"
    text: string
}

interface TestToolPart {
    type: "tool"
    tool: string
    input: unknown
    output?: string
    error?: string
}

type TestPart = TestTextPart | TestReasoningPart | TestToolPart

function modelLikeItem(item: Item): unknown[] {
    if (item.kind === "synthetic") return [{ type: "text", text: item.text }]
    const part = item.handle as TestPart
    if (part.type === "text") return [{ type: "text", text: part.text }]
    if (part.type === "reasoning") return [{ type: "reasoning", text: part.text }]
    if (part.type === "tool") return [toolShape(part)]
    return []
}

function toolShape(part: TestToolPart): unknown {
    if (part.error !== undefined) {
        return {
            type: `tool-${part.tool}`,
            state: "output-error",
            input: part.input,
            errorText: part.error,
        }
    }
    return {
        type: `tool-${part.tool}`,
        state: "output-available",
        input: part.input,
        output: part.output,
    }
}

const codec: CodecOps = {
    estimateTurns(turns) {
        const modelLike = turns
            .map((turn) => ({ role: turn.role, parts: turn.items.flatMap(modelLikeItem) }))
            .filter((entry) => entry.parts.length > 0)
        return countTokens(JSON.stringify(modelLike))
    },
    estimateItem(item) {
        return countTokens(JSON.stringify(toolShape(item.handle as TestToolPart)))
    },
    transcriptLine(item) {
        if (item.kind === "synthetic") return item.text
        const part = item.handle as TestPart
        if (part.type === "text") return part.text
        if (part.type === "reasoning") return `[reasoning]\n${part.text}`
        return `[tool:${part.tool}] input=${JSON.stringify(part.input)} output=${part.output ?? ""} error=${part.error ?? ""}`
    },
}

function toolHandle(item: Item): TestToolPart | null {
    return item.kind === "tool" ? (item.handle as TestToolPart) : null
}

const conventions: Conventions = {
    isSkillItem: (item) => toolHandle(item)?.tool === "skill",
    tool: (item) => {
        const tool = item.handle as TestToolPart
        return { name: tool.tool, input: tool.input, error: tool.error }
    },
    todo: {
        isTodoItem: (item) => toolHandle(item)?.tool === "todowrite",
        format: (item) => {
            const input = toolHandle(item)?.input as { todos: Array<Record<string, string>> }
            return input.todos
                .map(
                    (todo, index) =>
                        `${index + 1}. [${todo.status}/${todo.priority}] ${todo.content}`,
                )
                .join("; ")
        },
    },
}

const spec: LadderSpec = {
    codec,
    conventions,
    stages: [
        skillsStage,
        supersedeReadsStage,
        purgeErrorInputsStage,
        toolsOldStage,
        reasoningStage,
        toolsRemainingStage,
        assistantRunsStage,
    ],
}

const sessionKey = "ses_boundary_context"

function inputs(options: Partial<BuildPlanInputs> = {}): BuildPlanInputs {
    return {
        sessionKey,
        citablePath: (key, hash) => `.opencode/better-compact/sessions/${key}/${hash}.md`,
        ...options,
    }
}

function textItem(turnKey: string, text: string): Item {
    return {
        kind: "text",
        key: `${turnKey}-part`,
        text,
        handle: { type: "text", text } satisfies TestPart,
    }
}

function reasoningItem(turnKey: string, text: string): Item {
    return {
        kind: "reasoning",
        key: `${turnKey}-reasoning`,
        handle: { type: "reasoning", text } satisfies TestPart,
    }
}

function toolItem(
    turnKey: string,
    tool: string,
    output: string,
    input?: Record<string, unknown>,
): Item {
    const resolvedInput =
        input ?? (tool === "skill" ? { name: "root-cause-debug" } : { filePath: "src/app.ts" })
    return {
        kind: "tool",
        key: `${turnKey}-${tool}`,
        callId: `${turnKey}-${tool}-call-${output.length}-${JSON.stringify(resolvedInput).length}`,
        handle: { type: "tool", tool, input: resolvedInput, output } satisfies TestPart,
    }
}

function errorToolItem(
    turnKey: string,
    tool: string,
    error: string,
    input: Record<string, unknown>,
): Item {
    return {
        kind: "tool",
        key: `${turnKey}-${tool}`,
        callId: `${turnKey}-${tool}-call`,
        handle: { type: "tool", tool, input, error } satisfies TestPart,
    }
}

function turn(key: string, role: "user" | "assistant", items: Item[], stamp: number): Turn {
    return { key, stamp, role, items, handle: { key } }
}

function syntheticTextOf(target: Turn | undefined): string {
    assert.ok(target)
    return target.items
        .filter(
            (item): item is Extract<Item, { kind: "text" | "synthetic" }> =>
                item.kind === "text" || item.kind === "synthetic",
        )
        .map((item) => item.text)
        .join("\n")
}

function buildLargeConversation(): Turn[] {
    const bigToolOutput = "tool-output ".repeat(4_000)
    return [
        turn(
            "msg-user-1",
            "user",
            [textItem("msg-user-1", "Please preserve this exact requirement.")],
            1,
        ),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                reasoningItem("msg-assistant-1", "private reasoning ".repeat(2_000)),
                textItem("msg-assistant-1", "Investigated the OpenCode compaction path."),
                toolItem("msg-assistant-1", "read", bigToolOutput),
                toolItem("msg-assistant-1", "skill", "skill content ".repeat(2_000)),
            ],
            2,
        ),
        turn(
            "msg-user-2",
            "user",
            [textItem("msg-user-2", "Continue with the plugin-only design.")],
            3,
        ),
        turn(
            "msg-assistant-2",
            "assistant",
            [textItem("msg-assistant-2", "Recent assistant tail should remain raw.")],
            4,
        ),
        turn(
            "msg-user-3",
            "user",
            [textItem("msg-user-3", "Latest user tail should remain raw.")],
            5,
        ),
    ]
}

function buildMultiRunConversation(): Turn[] {
    return [
        turn(
            "msg-user-1",
            "user",
            [textItem("msg-user-1", "First task, keep this requirement.")],
            1,
        ),
        turn(
            "msg-assistant-big",
            "assistant",
            [
                reasoningItem("msg-assistant-big", "big private reasoning ".repeat(500)),
                textItem("msg-assistant-big", "big assistant detail ".repeat(7_000)),
                toolItem("msg-assistant-big", "read", "big tool output ".repeat(500)),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "Second task.")], 3),
        turn(
            "msg-assistant-small",
            "assistant",
            [
                reasoningItem("msg-assistant-small", "small private reasoning ".repeat(200)),
                textItem("msg-assistant-small", "small assistant detail"),
                toolItem("msg-assistant-small", "grep", "small tool output ".repeat(200), {
                    pattern: "needle",
                }),
            ],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "Third task stays raw.")], 5),
        turn(
            "msg-assistant-tail",
            "assistant",
            [textItem("msg-assistant-tail", "tail assistant")],
            6,
        ),
        turn("msg-user-4", "user", [textItem("msg-user-4", "Latest user stays raw.")], 7),
    ]
}

function buildReferenceIndexConversation(): Turn[] {
    const longTodo = "follow-up ".repeat(70).trim()
    return [
        turn("msg-user-1", "user", [textItem("msg-user-1", "Implement the parser change.")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                textItem(
                    "msg-assistant-1",
                    "Implemented the parser change. Follow-up narration.\nExtra details.",
                ),
                toolItem("msg-assistant-1", "read", "source", { filePath: "src/./parser.ts" }),
            ],
            2,
        ),
        turn(
            "msg-assistant-1b",
            "assistant",
            [toolItem("msg-assistant-1b", "bash", "passing", { command: "pnpm test" })],
            3,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "Verify the edge case.")], 4),
        turn(
            "msg-assistant-2",
            "assistant",
            [
                textItem("msg-assistant-2", "Verified the edge case."),
                toolItem("msg-assistant-2", "grep", "match", { pattern: "needle" }),
                toolItem("msg-assistant-2", "todowrite", "saved", {
                    todos: [
                        { content: "current task", status: "in_progress", priority: "high" },
                        { content: longTodo, status: "pending", priority: "medium" },
                    ],
                }),
            ],
            5,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "Keep this raw.")], 6),
        turn(
            "msg-assistant-tail",
            "assistant",
            [textItem("msg-assistant-tail", "Raw assistant tail.")],
            7,
        ),
        turn("msg-user-4", "user", [textItem("msg-user-4", "Latest user turn.")], 8),
    ]
}

test("planner does nothing before 85 percent usage", () => {
    const plan = buildPlan(
        [turn("msg-user-small", "user", [textItem("msg-user-small", "small")], 1)],
        inputs({ contextLimit: 100_000 }),
        spec,
    )

    assert.equal(plan, null)
})

test("planner compactifies old assistant/tool context and preserves raw tail", () => {
    const turns = buildLargeConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )

    assert.ok(plan)
    assert.ok(plan.afterPruneTokens < plan.beforeTokens)
    assert.ok(plan.stages.some((stage) => stage.name === "reasoning" && stage.clearedTokens > 0))
    assert.ok(plan.stages.some((stage) => stage.name === "skills" && stage.clearedTokens > 0))
    assert.ok(plan.stages.some((stage) => stage.name === "tools-old" && stage.clearedTokens > 0))
    assert.match(
        plan.transcript.relativePath,
        /^\.opencode\/better-compact\/sessions\/ses_boundary_context\//,
    )
    assert.equal(plan.transcript.content, "")
    assert.deepEqual(plan.transcript.messageIds, ["msg-user-1", "msg-assistant-1"])

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    assert.equal(transformed.at(-1)?.key, "msg-user-3")
    assert.equal(transformed.at(-2)?.key, "msg-assistant-2")
    assert.equal(transformed.at(-3)?.key, "msg-user-2")

    const firstUserItem = transformed[0]?.items[0]
    assert.equal(firstUserItem?.kind, "text")
    if (firstUserItem?.kind === "text") {
        assert.equal(firstUserItem.text, "Please preserve this exact requirement.")
    }

    const compactedText = syntheticTextOf(transformed[1])
    assert.match(compactedText, /Investigated the OpenCode compaction path/)
    assert.doesNotMatch(compactedText, /private reasoning/)
    assert.doesNotMatch(compactedText, /skill content/)
    assert.doesNotMatch(compactedText, /tool-output/)

    const reference = transformed.find((item) => item.key.startsWith("better_compact_context_"))
    assert.ok(reference)
    const referenceText = syntheticTextOf(reference)
    assert.match(referenceText, /## Reference Files/)
    assert.match(
        referenceText,
        new RegExp(plan.transcript.relativePath.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")),
    )
})

test("reference turn indexes every compacted assistant run and surfaces the latest todo", () => {
    const turns = buildReferenceIndexConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 1_000_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const referenceIndex = transformed.findIndex((item) =>
        item.key.startsWith("better_compact_context_"),
    )
    assert.ok(referenceIndex >= 0)
    assert.equal(transformed[referenceIndex + 1]?.key, "msg-user-3")

    const referenceText = syntheticTextOf(transformed[referenceIndex])
    const runLines = referenceText.split("\n").filter((line) => line.startsWith("- msg-assistant"))
    // Each assistant turn is indexed on its own line, so a paid summary and
    // its tool calls stay attached to the turn they came from.
    assert.deepEqual(runLines, [
        "- msg-assistant-1 — read src/parser.ts — Implemented the parser change.",
        "- msg-assistant-1b — bash pnpm test — (no assistant text)",
        "- msg-assistant-2 — grep needle, todowrite — Verified the edge case.",
    ])
    assert.equal(
        referenceText.split("\n").at(-1),
        `Latest todo state preserved: 1. [in_progress/high] current task; 2. [pending/medium] ${"follow-up ".repeat(70).trim()}`,
    )
})

test("reference index replays byte-stably", () => {
    const turns = buildReferenceIndexConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 1_000_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const replayed = replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true })

    assert.ok(replayed)
    assert.equal(JSON.stringify(replayed), JSON.stringify(transformed))
})

test("catalog-backed handoff omits per-turn transcript paths and replays stably", () => {
    const turns = buildReferenceIndexConversation()
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 1_000_000,
            force: true,
            recentToolResultBudgetTokens: 0,
            archiveCatalogText: "- c000001-123456789abc — Earlier parser work",
        }),
        spec,
    )
    assert.ok(plan)
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const live = applied.map((turn) => syntheticTextOf(turn)).join("\n")
    assert.match(live, /Earlier parser work/)
    assert.match(live, /better_compact_recall/)
    assert.doesNotMatch(live, /Raw transcript:|## Reference Files/)
    assert.doesNotMatch(live, /\.opencode\/better-compact\/sessions\//)
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true }),
        applied,
    )
})

test("applied output matches the simulated plan when assistant runs are summarized", () => {
    const turns = buildMultiRunConversation()
    const options = inputs({ contextLimit: 40_000, recentToolResultBudgetTokens: 0 })
    const firstPass = buildPlan(turns, options, spec)
    assert.ok(firstPass)
    assert.ok(
        firstPass.stages.some(
            (stage) => stage.name === "assistant-runs" && stage.status === "applied",
        ),
    )
    assert.ok(firstPass.summaryJobs.length > 0)

    const assistantSummaries = Object.fromEntries(
        firstPass.assistantSummaryKeys.map((key) => [
            key,
            "Accepted summary: shipped the first task end to end.",
        ]),
    )
    const plan = buildPlan(turns, { ...options, assistantSummaries }, spec)
    assert.ok(plan)
    assert.equal(plan.summaryJobs.length, 0)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    assert.equal(plan.afterPruneTokens, codec.estimateTurns(transformed))
    assert.ok(!plan.stages.some((stage) => stage.name === "prefix-summary"))
    assert.ok(codec.estimateTurns(transformed) < plan.triggerTokens)

    const selectedRun = transformed.find((item) => item.key === "msg-assistant-big")
    assert.match(
        syntheticTextOf(selectedRun),
        /Accepted summary: shipped the first task end to end\./,
    )

    // The core drift bug: non-selected prefix runs must keep the stage 1-4
    // pruning (no tool/reasoning items) in the applied output.
    const nonSelectedRun = transformed.find((item) => item.key === "msg-assistant-small")
    assert.ok(nonSelectedRun)
    assert.ok(
        !nonSelectedRun.items.some((item) => item.kind === "tool" || item.kind === "reasoning"),
    )

    assert.ok(transformed.some((item) => item.key === "msg-assistant-tail"))
})

test("separate grouped calls rebuild and replay summaries in original turn order", async () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "First request")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [textItem("msg-assistant-1", "old first ".repeat(3_000))],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "Second request")], 3),
        turn(
            "msg-assistant-2",
            "assistant",
            [textItem("msg-assistant-2", "old second ".repeat(3_000))],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "Third request")], 5),
        turn(
            "msg-assistant-3",
            "assistant",
            [textItem("msg-assistant-3", "old third ".repeat(3_000))],
            6,
        ),
        turn("msg-user-4", "user", [textItem("msg-user-4", "Last request")], 7),
        turn("msg-assistant-4", "assistant", [textItem("msg-assistant-4", "recent reply")], 8),
        turn("msg-user-5", "user", [textItem("msg-user-5", "Latest request")], 9),
    ]
    const original = JSON.stringify(turns)
    const options = inputs({
        contextLimit: 50_000,
        targetRatio: 0.01,
        force: true,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: false,
    })
    const initial = buildPlan(turns, options, spec)
    assert.ok(initial)
    assert.ok(initial.summaryJobs.length >= 3)

    const calls: string[][] = []
    const scheduler = createSummaryScheduler({ info() {}, debug() {}, warn() {}, error() {} })
    const summaries = await scheduler.summarize({
        sessionKey,
        jobs: initial.summaryJobs,
        concurrency: 2,
        maxCalls: 2,
        targetBatchTokens: 1,
        maxJobsPerBatch: 2,
        summarizer: {
            complete: async () => {
                throw new Error("Must use grouped transport")
            },
            completeBatch: async (batch) => {
                calls.push(batch.map((job) => job.key))
                return Object.fromEntries(
                    batch
                        .filter((job) => job.rangeStartMessageId !== "msg-assistant-2")
                        .map((job) => [
                            job.key,
                            [
                                "## Decisions",
                                `- Accepted ${job.rangeStartMessageId} and kept its decisions.`,
                                "## Files & Symbols",
                                "- src/feature.ts",
                                "## Errors (verbatim)",
                                "- (none)",
                                "## What failed and why",
                                "- (none)",
                                "## Constraints",
                                "- Preserve the user request.",
                                "## Next step",
                                "- Continue the work.",
                            ].join("\n"),
                        ]),
                )
            },
        },
    })
    assert.equal(calls.length, 2)
    assert.ok(calls.some((batch) => batch.length === 2))
    assert.equal(Object.keys(summaries).length, 2)
    const rebuilt = buildPlan(
        turns,
        { ...options, assistantSummaries: summaries, priorPlan: toPlanSnapshot(initial) },
        spec,
    )
    assert.ok(rebuilt)
    const transformed = transformTurns(turns, rebuilt.rawTailStartIndex, rebuilt, spec)
    const replay = replayPlanSnapshot(turns, toPlanSnapshot(rebuilt), spec, { allowRegrown: true })
    assert.ok(replay)
    assert.equal(JSON.stringify(replay), JSON.stringify(transformed))

    for (const accepted of initial.summaryJobs.filter((job) => summaries[job.key])) {
        const position = transformed.findIndex((item) => item.key === accepted.rangeStartMessageId)
        assert.ok(position >= 0)
        const content = syntheticTextOf(transformed[position])
        assert.match(content, new RegExp(`Accepted ${accepted.rangeStartMessageId}`))
        for (const other of initial.summaryJobs.filter(
            (job) => job.key !== accepted.key && summaries[job.key],
        )) {
            assert.doesNotMatch(content, new RegExp(`Accepted ${other.rangeStartMessageId}`))
        }
    }
    const skipped = initial.summaryJobs.find((job) => !summaries[job.key])
    assert.ok(skipped)
    const fallback = transformed.find((item) => item.key === skipped.rangeStartMessageId)
    assert.match(syntheticTextOf(fallback), /old (first|second|third)/)
    assert.equal(JSON.stringify(turns), original)
})

test("prefix summary fires when pruning cannot get the applied output below trigger", () => {
    const turns = buildMultiRunConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 500, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.requiresCustomCompaction, true)
    assert.ok(plan.stages.some((stage) => stage.name === "prefix-summary"))

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const summary = syntheticTextOf(transformed[0])

    assert.equal(plan.afterPruneTokens, codec.estimateTurns(transformed))
    assert.ok(transformed[0]?.key.startsWith("better_compact_summary_"))
    assert.match(
        summary,
        /## Decisions[\s\S]*## Files & Symbols[\s\S]*## Errors \(verbatim\)[\s\S]*## What failed and why[\s\S]*## Constraints[\s\S]*## Next step/,
    )
    assert.equal(transformed.at(-1)?.key, "msg-user-4")
})

test("an already-triggered pass chases the target even after cheap pruning falls below the trigger", () => {
    const turns = buildMultiRunConversation()
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 40_000,
            force: true,
            triggerTokens: 35_000,
            targetTokens: 100,
            recentToolResultBudgetTokens: 0,
        }),
        spec,
    )
    assert.ok(plan)
    const prefix = plan.stages.find((stage) => stage.name === "prefix-summary")
    assert.ok(prefix, "the target, not the trigger, must gate last-resort synthesis")
    assert.ok(prefix.beforeTokens < plan.triggerTokens)
    assert.ok(prefix.beforeTokens > plan.targetTokens)
})

test("best-effort target does not launch a new last resort within fifteen percent", () => {
    const turns = buildMultiRunConversation()
    const simpleSpec = { ...spec, stages: [toolsOldStage, reasoningStage] }
    const base = inputs({
        contextLimit: 40_000,
        force: true,
        minTailUserTurns: 1,
        recentToolResultBudgetTokens: 0,
        targetTokens: 100,
        prefixSummaryAllowed: false,
    })
    const cheap = buildPlan(turns, base, simpleSpec)
    assert.ok(cheap)
    const withinTarget = Math.ceil(cheap.afterPruneTokens / 1.1)
    const within = buildPlan(
        turns,
        { ...base, targetTokens: withinTarget, prefixSummaryAllowed: true },
        simpleSpec,
    )
    assert.ok(within)
    assert.ok(within.afterPruneTokens > within.targetTokens)
    assert.ok(within.afterPruneTokens <= Math.floor(within.targetTokens * 1.15))
    assert.equal(within.requiresCustomCompaction, false)
    assert.equal(
        within.stages.some((stage) => stage.name === "prefix-summary"),
        false,
    )

    const outside = buildPlan(
        turns,
        {
            ...base,
            targetTokens: Math.floor(cheap.afterPruneTokens / 1.2),
            prefixSummaryAllowed: true,
        },
        simpleSpec,
    )
    assert.ok(outside?.requiresCustomCompaction)
})

test("OpenCode last resort fills available context with newest native prefix turns", () => {
    const turns = [
        turn("first-user", "user", [textItem("first-user", "Keep the original request")], 1),
        ...Array.from({ length: 12 }, (_, index) =>
            turn(
                `progress-${index}`,
                "assistant",
                [
                    textItem(
                        `progress-${index}`,
                        `Step ${index}: ${"specific evidence ".repeat(160)}`,
                    ),
                ],
                index + 2,
            ),
        ),
        turn(
            "progress-huge",
            "assistant",
            [
                textItem(
                    "progress-huge",
                    `Oversized older result: ${"bulky evidence ".repeat(5_000)}`,
                ),
            ],
            19,
        ),
        turn("latest-user", "user", [textItem("latest-user", "Continue the task")], 20),
        turn("latest-assistant", "assistant", [textItem("latest-assistant", "Current work")], 21),
    ]
    const config = inputs({
        contextLimit: 50_000,
        force: true,
        targetTokens: 3_000,
        collapsePercent: 10,
        minTailUserTurns: 1,
        recentToolResultBudgetTokens: 0,
        preservePrefixBudgets: true,
    })
    const plan = buildPlan(turns, config, spec)
    assert.ok(plan?.requiresCustomCompaction)
    assert.ok(plan.preservedPrefixTurnKeys?.length)
    assert.ok(
        plan.preservedPrefixTurnKeys.some(
            (key) => key.startsWith("progress-") && key !== "progress-huge",
        ),
    )
    assert.ok(!plan.preservedPrefixTurnKeys.includes("progress-huge"))
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const barePlan = buildPlan(turns, { ...config, preservePrefixBudgets: false }, spec)
    assert.ok(barePlan)
    assert.ok(plan.afterPruneTokens > barePlan.afterPruneTokens + 500)
    assert.ok(plan.afterPruneTokens <= plan.targetTokens)
    assert.ok(
        plan.afterPruneTokens >= plan.targetTokens * 0.8,
        `native turns should use available headroom: ${plan.afterPruneTokens} vs ${plan.targetTokens}`,
    )
    assert.equal(applied[0].role, "user")
    assert.equal(plan.afterPruneTokens, codec.estimateTurns(applied))
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true }),
        applied,
    )
    assert.equal(applied.at(-1)?.key, "latest-assistant")

    const continued = [
        ...turns,
        turn("next-user", "user", [textItem("next-user", "Another correction")], 22),
        turn("next-assistant", "assistant", [textItem("next-assistant", "Continue work")], 23),
    ]
    const next = buildPlan(continued, { ...config, priorPlan: toPlanSnapshot(plan) }, spec)
    assert.ok(next?.requiresCustomCompaction)
    for (const key of plan.preservedPrefixTurnKeys ?? []) {
        assert.ok(next.preservedPrefixTurnKeys?.includes(key), `previously live ${key} disappeared`)
    }
    assert.deepEqual(
        replayPlanSnapshot(continued, toPlanSnapshot(next), spec, { allowRegrown: true }),
        transformTurns(continued, next.rawTailStartIndex, next, spec),
    )
})

test("OpenCode cheap tool pruning retains newer whole results rather than overshooting the target", () => {
    const turns = [turn("floor-first", "user", [textItem("floor-first", "Preserve the task")], 1)]
    turns.push(
        turn(
            "floor-huge",
            "assistant",
            [
                toolItem("floor-huge", "bash", "large archived output ".repeat(4_000), {
                    command: "earlier-heavy-job",
                }),
            ],
            2,
        ),
    )
    for (let index = 0; index < 12; index++) {
        const key = `floor-tool-${index}`
        turns.push(
            turn(
                key,
                "assistant",
                [
                    toolItem(key, "bash", `Useful check ${index}: ${"result ".repeat(180)}`, {
                        command: `check-${index}`,
                    }),
                ],
                3 + index,
            ),
        )
    }
    turns.push(
        turn("floor-current", "user", [textItem("floor-current", "Current task stays raw")], 20),
    )
    const options = inputs({
        contextLimit: 50_000,
        force: true,
        targetTokens: 1_000,
        minTailUserTurns: 1,
        recentToolResultBudgetTokens: 0,
        preservePrefixBudgets: true,
        prefixSummaryAllowed: false,
        recentAssistantOutputs: 0,
        archiveCatalogText: "- c000001 — Exact actions",
    })
    const baseline = buildPlan(turns, { ...options, preservePrefixBudgets: false }, spec)
    const plan = buildPlan(turns, options, spec)
    assert.ok(baseline && plan)
    assert.ok(baseline.afterPruneTokens < options.targetTokens!)
    assert.ok(plan.afterPruneTokens > baseline.afterPruneTokens)
    assert.ok(plan.afterPruneTokens <= plan.targetTokens)
    assert.ok(plan.preservedToolCallIds.length > 0)
    assert.ok(
        !plan.preservedToolCallIds.includes(
            (turns[1].items[0] as Extract<Item, { kind: "tool" }>).callId,
        ),
    )
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.ok(applied.some((turn) => turn.items.some((item) => item.kind === "tool")))
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true }),
        applied,
    )
})

test("a validated round-two retirement can replace old native turns without dropping the latest five outputs", () => {
    const turns = [turn("first-user", "user", [textItem("first-user", "Keep violet widgets")], 1)]
    for (let index = 0; index < 12; index++)
        turns.push(
            turn(
                `old-output-${index}`,
                "assistant",
                [
                    textItem(
                        `old-output-${index}`,
                        `Earlier result ${index}: ${"source evidence ".repeat(170)}`,
                    ),
                ],
                index + 2,
            ),
        )
    turns.push(turn("user-current", "user", [textItem("user-current", "Apply violet rule")], 20))
    const config = inputs({
        contextLimit: 50_000,
        targetTokens: 3_000,
        minTailUserTurns: 1,
        recentToolResultBudgetTokens: 0,
        collapsePercent: 10,
        preservePrefixBudgets: true,
        force: true,
    })
    const first = buildPlan(turns, config, spec)
    assert.ok(first?.requiresCustomCompaction)
    assert.ok(first.preservedPrefixTurnKeys?.length)
    const continued = [
        ...turns,
        ...Array.from({ length: 8 }, (_, index) =>
            turn(
                `new-output-${index}`,
                "assistant",
                [
                    textItem(
                        `new-output-${index}`,
                        `New answer ${index}: ${"current evidence ".repeat(170)}`,
                    ),
                ],
                index + 21,
            ),
        ),
        turn("new-user", "user", [textItem("new-user", "Keep the latest violet correction")], 30),
    ]
    const handoff =
        "## Decisions\n- Violet widgets remain required.\n## Next step\n- Continue the current correction."
    const nextConfig = { ...config, recentAssistantOutputs: 5 }
    const stillCarrying = buildPlan(
        continued,
        { ...nextConfig, priorPlan: toPlanSnapshot(first), prefixSummary: handoff },
        spec,
    )
    const retired = buildPlan(
        continued,
        {
            ...nextConfig,
            priorPlan: toPlanSnapshot(first),
            prefixSummary: handoff,
            retirementThrough: 1,
        },
        spec,
    )
    assert.ok(stillCarrying?.requiresCustomCompaction)
    assert.ok(retired?.requiresCustomCompaction)
    assert.ok(
        retired.afterPruneTokens < stillCarrying.afterPruneTokens,
        "the accepted handoff should replace associated old native turns, not coexist with them",
    )
    for (const key of first.preservedPrefixTurnKeys ?? []) {
        assert.ok(!retired.preservedPrefixTurnKeys?.includes(key), `retired ${key} is still native`)
        assert.ok(
            retired.transcript.turns?.some((turn) => turn.key === key),
            `missing exact ${key} from archive source`,
        )
    }
    assert.equal(retired.retirementThrough, 1)
    const applied = transformTurns(continued, retired.rawTailStartIndex, retired, spec)
    for (let index = 3; index < 8; index++) {
        const output = applied.find((turn) => turn.key === `new-output-${index}`)?.items[0]
        assert.ok(output?.kind === "text", `current assistant output ${index} was lost`)
        assert.equal(
            output.text,
            `New answer ${index}: ${"current evidence ".repeat(170)}`,
            `current assistant output ${index} did not survive byte-exact`,
        )
    }
    assert.deepEqual(
        replayPlanSnapshot(continued, toPlanSnapshot(retired), spec, { allowRegrown: true }),
        applied,
    )
})

test("five latest real assistant outputs keep their entire reasoning span without tool calls", () => {
    const turns = [turn("start", "user", [textItem("start", "Investigate")], 1)]
    for (let index = 0; index < 7; index++) {
        turns.push(
            turn(
                `output-${index}`,
                "assistant",
                [
                    reasoningItem(`output-${index}`, `Reasoning ${index} ${"step ".repeat(500)}`),
                    textItem(`output-${index}`, `Actual answer ${index}`),
                    toolItem(`output-${index}`, "read", `Raw tool data ${index}`),
                ],
                index * 2 + 2,
            ),
            turn(
                `tool-only-${index}`,
                "assistant",
                [toolItem(`tool-only-${index}`, "read", `More tool data ${index}`)],
                index * 2 + 3,
            ),
        )
    }
    turns.push(turn("latest", "user", [textItem("latest", "Continue")], 30))
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 50_000,
            force: true,
            minTailUserTurns: 1,
            targetTokens: 2_000,
            recentToolResultBudgetTokens: 0,
            recentReasoningBudgetTokens: 100,
            recentAssistantOutputs: 5,
            preservePrefixBudgets: true,
        }),
        spec,
    )
    assert.ok(plan?.requiresCustomCompaction)
    assert.equal(plan.anchorReasoningLimited, undefined)
    assert.deepEqual(
        plan.protectedAssistantItemKeys,
        [2, 3, 4, 5, 6].map((index) => `output-${index}-part`),
    )
    assert.deepEqual(
        new Set(plan.preservedReasoningItemKeys),
        new Set([2, 3, 4, 5, 6].map((index) => `output-${index}-reasoning`)),
    )
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    for (let index = 2; index < 7; index++) {
        const native = applied.find((item) => item.key === `output-${index}`)
        assert.ok(native)
        assert.ok(native.items.some((item) => item.kind === "text"))
        assert.ok(native.items.some((item) => item.kind === "reasoning"))
        assert.ok(native.items.every((item) => item.kind !== "tool"))
    }
    assert.equal(applied.at(-1)?.key, "latest")
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true }),
        applied,
    )
})

test("an oversized five-output reasoning span keeps answers and bounds thoughts with a buffer", () => {
    const turns = [turn("start", "user", [textItem("start", "Investigate")], 1)]
    for (let index = 0; index < 5; index++) {
        turns.push(
            turn(
                `answer-${index}`,
                "assistant",
                [
                    reasoningItem(`answer-${index}`, `Reasoning ${index} ${"step ".repeat(3_000)}`),
                    textItem(`answer-${index}`, `Output ${index}`),
                ],
                index + 2,
            ),
        )
    }
    turns.push(turn("latest", "user", [textItem("latest", "Continue")], 8))
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 12_000,
            force: true,
            minTailUserTurns: 1,
            targetTokens: 2_000,
            recentToolResultBudgetTokens: 0,
            recentReasoningBudgetTokens: 500,
            recentAssistantOutputs: 5,
            preservePrefixBudgets: true,
        }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.anchorReasoningLimited, true)
    assert.equal(plan.protectedAssistantItemKeys?.length, 5)
    assert.equal(plan.preservedReasoningItemKeys?.length, 0)
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.ok(codec.estimateTurns(applied) < plan.contextLimit)
    for (let index = 0; index < 5; index++) {
        assert.ok(
            applied
                .find((item) => item.key === `answer-${index}`)
                ?.items.some((item) => item.kind === "text"),
        )
    }
})

test("projection does not scale transformed context by raw provider ratio", () => {
    const turns = buildLargeConversation()
    const providerReportedTokens = 10_000
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 100_000,
            force: true,
            providerReportedTokens,
            recentToolResultBudgetTokens: 0,
        }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const directAfter = codec.estimateTurns(transformed)
    const oldScaledAfter = Math.round(
        directAfter * (providerReportedTokens / codec.estimateTurns(turns)),
    )

    assert.equal(plan.beforeTokens, providerReportedTokens)
    assert.equal(plan.afterPruneTokens, directAfter + plan.overheadTokens)
    assert.ok(plan.afterPruneTokens > oldScaledAfter * 2)
})

function storedPlanFor(turns: Turn[]) {
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 40_000, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    return toPlanSnapshot(plan)
}

test("plan snapshot refuses to apply when the prefix was edited", () => {
    const snapshot = storedPlanFor(buildMultiRunConversation())

    const replayed = replayPlanSnapshot(buildMultiRunConversation(), snapshot, spec)
    assert.ok(replayed)
    assert.ok(replayed.some((item) => item.key.startsWith("better_compact_context_")))

    const edited = buildMultiRunConversation()
    edited[1].stamp = 999
    assert.equal(replayPlanSnapshot(edited, snapshot, spec), null)
})

test("plan snapshot refuses to apply once the transformed output regrows past trigger", () => {
    const snapshot = storedPlanFor(buildMultiRunConversation())

    const regrown = buildMultiRunConversation()
    for (let index = 0; index < 12; index++) {
        regrown.push(
            turn(
                `msg-user-new-${index}`,
                "user",
                [textItem(`msg-user-new-${index}`, "next task")],
                100 + index * 2,
            ),
            turn(
                `msg-assistant-new-${index}`,
                "assistant",
                [textItem(`msg-assistant-new-${index}`, "fresh assistant output ".repeat(2_000))],
                101 + index * 2,
            ),
        )
    }
    assert.equal(replayPlanSnapshot(regrown, snapshot, spec), null)
})

test("provider-reported totals keep plan accounting on a single scale", () => {
    const turns = buildMultiRunConversation()
    const rawEstimate = codec.estimateTurns(turns)
    const providerReportedTokens = rawEstimate + 50_000
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 120_000,
            force: true,
            recentToolResultBudgetTokens: 0,
            providerReportedTokens,
        }),
        spec,
    )
    assert.ok(plan)

    assert.equal(plan.beforeTokens, providerReportedTokens)
    assert.equal(plan.overheadTokens, providerReportedTokens - rawEstimate)
    assert.ok(plan.afterPruneTokens >= plan.overheadTokens)
    assert.ok(plan.beforeTokens - plan.afterPruneTokens >= 0)
    for (const stage of plan.stages) {
        assert.ok(stage.beforeTokens >= plan.overheadTokens)
        assert.ok(stage.afterTokens >= plan.overheadTokens)
        assert.ok(stage.clearedTokens <= plan.beforeTokens)
    }

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.equal(plan.afterPruneTokens, codec.estimateTurns(transformed) + plan.overheadTokens)
})

test("fresh tool content cannot erase overhead measured against the previous provider history", () => {
    const turns = buildMultiRunConversation()
    const rawCurrent = codec.estimateTurns(turns)
    const previousProviderTotal = rawCurrent + 1_000
    const priorHistoryWithOutput = Math.floor(rawCurrent / 2)
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 120_000,
            force: true,
            recentToolResultBudgetTokens: 0,
            providerReportedTokens: previousProviderTotal,
            providerHistoryTokens: priorHistoryWithOutput,
        }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.overheadTokens, previousProviderTotal - priorHistoryWithOutput)
    assert.equal(
        plan.afterPruneTokens,
        codec.estimateTurns(transformTurns(turns, plan.rawTailStartIndex, plan, spec)) +
            plan.overheadTokens,
    )
})

test("planner marks custom compaction as last resort only after pruning is still too large", () => {
    const plan = buildPlan(
        buildLargeConversation(),
        inputs({ contextLimit: 200, recentToolResultBudgetTokens: 0 }),
        spec,
    )

    assert.ok(plan)
    assert.equal(plan.requiresCustomCompaction, true)
})

test("last-resort summary emits exactly one transcript reference section", () => {
    const turns = buildLargeConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 200, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const summary = syntheticTextOf(transformed[0])
    const referenceBlock = `## Reference Files\n- "${plan.transcript.relativePath}"`

    assert.equal(summary.match(/## Reference Files/g)?.length, 1)
    assert.equal(summary.split(referenceBlock).length - 1, 1)
})

test("legacy snapshot summary does not duplicate its persisted transcript reference section", () => {
    const turns = buildLargeConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 200, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    const snapshot = toPlanSnapshot(plan)
    const referenceBlock = `## Reference Files\n- "${snapshot.transcriptRelativePath}"`
    snapshot.prefixSummary = `${snapshot.prefixSummary}\n\n${referenceBlock}`

    const replayed = replayPlanSnapshot(turns, snapshot, spec, { allowRegrown: true })
    assert.ok(replayed)
    const summary = syntheticTextOf(replayed[0])

    assert.equal(summary.match(/## Reference Files/g)?.length, 1)
    assert.equal(summary.split(referenceBlock).length - 1, 1)
})

test("replacement plan replaces a legacy transcript reference from its prior plan", () => {
    const turns = buildLargeConversation()
    const first = buildPlan(
        turns,
        inputs({ contextLimit: 200, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(first)
    const prior = toPlanSnapshot(first)
    const priorReferenceBlock = `## Reference Files\n- "${prior.transcriptRelativePath}"`
    prior.prefixSummary = `${prior.prefixSummary}\n\n${priorReferenceBlock}`

    const replacement = buildPlan(
        turns,
        inputs({
            sessionKey: "forked-session",
            contextLimit: 200,
            recentToolResultBudgetTokens: 0,
            priorPlan: prior,
        }),
        spec,
    )
    assert.ok(replacement)
    assert.notEqual(replacement.transcript.relativePath, prior.transcriptRelativePath)

    const transformed = transformTurns(turns, replacement.rawTailStartIndex, replacement, spec)
    const summary = syntheticTextOf(transformed[0])
    const replacementReferenceBlock = `## Reference Files\n- "${replacement.transcript.relativePath}"`

    assert.equal(summary.match(/## Reference Files/g)?.length, 1)
    assert.equal(summary.includes(priorReferenceBlock), false)
    assert.equal(summary.split(replacementReferenceBlock).length - 1, 1)
})

test("planner preserves the latest two user turns as raw tail", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old user")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [toolItem("msg-assistant-1", "read", "old output ".repeat(4_000))],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user must stay raw")], 3),
        turn(
            "msg-assistant-2",
            "assistant",
            [textItem("msg-assistant-2", "middle assistant must stay raw")],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user must stay raw")], 5),
        turn(
            "msg-assistant-3",
            "assistant",
            [textItem("msg-assistant-3", "latest assistant must stay raw")],
            6,
        ),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    assert.ok(transformed.some((item) => item.key === "msg-user-2"))
    assert.ok(transformed.some((item) => item.key === "msg-assistant-2"))
    assert.ok(transformed.some((item) => item.key === "msg-user-3"))
    assert.ok(transformed.some((item) => item.key === "msg-assistant-3"))
    const oldAssistantText = syntheticTextOf(
        transformed.find((item) => item.key === "msg-assistant-1"),
    )
    assert.match(oldAssistantText, /\[tool:read\] src\/app\.ts — ok|Assistant turn summary/)
    assert.doesNotMatch(oldAssistantText, /old output/)
})

test("planner preserves the active prompt during a single-user tool loop", () => {
    const activePrompt = "Read the lockfile completely and report its architecture."
    const turns = [
        turn(
            "msg-user-active",
            "user",
            [
                textItem("msg-user-reminder", "global system reminder ".repeat(2_000)),
                textItem("msg-user-prompt", activePrompt),
            ],
            1,
        ),
        turn(
            "msg-assistant-1",
            "assistant",
            [toolItem("msg-assistant-1", "read", "first tool output ".repeat(4_000))],
            2,
        ),
        turn(
            "msg-assistant-2",
            "assistant",
            [toolItem("msg-assistant-2", "read", "second tool output ".repeat(4_000))],
            3,
        ),
        turn(
            "msg-assistant-3",
            "assistant",
            [toolItem("msg-assistant-3", "read", "third tool output ".repeat(4_000))],
            4,
        ),
    ]
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 10_000,
            triggerRatio: 0.03,
            targetRatio: 0.01,
            recentToolResultBudgetTokens: 0,
            force: true,
        }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    assert.ok(plan.afterPruneTokens < plan.beforeTokens)
    assert.ok(
        plan.stages.some(
            (stage) =>
                (stage.name === "supersede-reads" ||
                    stage.name === "tools-old" ||
                    stage.name === "tools-remaining") &&
                stage.clearedTokens > 0,
        ),
    )
    const transformedText = transformed.map(syntheticTextOf).join("\n")
    assert.match(transformedText, new RegExp(activePrompt.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")))
    assert.doesNotMatch(transformedText, /first tool output|second tool output|third tool output/)
})

test("planner compactifies each contiguous assistant turn on its own", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [textItem("msg-assistant-1", "first assistant detail ".repeat(4_000))],
            2,
        ),
        turn(
            "msg-assistant-2",
            "assistant",
            [
                toolItem("msg-assistant-2", "bash", "build output ".repeat(4_000), {
                    command: "npm test",
                }),
            ],
            3,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 4),
        turn("msg-assistant-3", "assistant", [textItem("msg-assistant-3", "middle assistant")], 5),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user")], 6),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 5_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    assert.ok(plan.stages.some((stage) => stage.name === "assistant-runs"))
    assert.ok(plan.summaryJobs.length > 0)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    // One turn is one selection unit: a large turn is summarized without
    // dragging the turn beside it into the same summary, and each keeps its
    // own key so a paid summary lands on the turn it describes.
    const oldAssistantTurns = transformed.filter(
        (item) => item.key === "msg-assistant-1" || item.key === "msg-assistant-2",
    )
    assert.equal(oldAssistantTurns.length, 2)
    const compactedText = oldAssistantTurns.map(syntheticTextOf).join("\n")
    assert.match(compactedText, /first assistant detail/)
    // The call is still named — that is what a stub is for — while the tool's
    // bulk output is gone.
    assert.match(compactedText, /\[tool:bash\]/)
    assert.doesNotMatch(compactedText, /build output/)
})

test("planner ranks assistant turns and summarizes only enough to meet target", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old user")], 1),
        turn(
            "msg-assistant-big-old",
            "assistant",
            [textItem("msg-assistant-big-old", "big old detail ".repeat(8_000))],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        turn(
            "msg-assistant-small-old",
            "assistant",
            [textItem("msg-assistant-small-old", "small old detail ".repeat(200))],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "newer user")], 5),
        turn(
            "msg-assistant-big-newer",
            "assistant",
            [textItem("msg-assistant-big-newer", "big newer detail ".repeat(8_000))],
            6,
        ),
        turn("msg-user-4", "user", [textItem("msg-user-4", "tail user")], 7),
        turn(
            "msg-assistant-tail",
            "assistant",
            [textItem("msg-assistant-tail", "tail assistant")],
            8,
        ),
        turn("msg-user-5", "user", [textItem("msg-user-5", "latest user")], 9),
    ]

    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 50_000,
            force: true,
            targetRatio: 0.7,
            recentToolResultBudgetTokens: 0,
        }),
        spec,
    )

    assert.ok(plan)
    assert.ok(plan.summaryJobs.length > 0)
    assert.ok(plan.summaryJobs.length < 3)
    assert.match(plan.summaryJobs[0].rangeStartMessageId, /msg-assistant-big/)
})

test("planner preserves recent tool results under the tool-tail budget", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old user")], 1),
        turn(
            "msg-assistant-old",
            "assistant",
            [toolItem("msg-assistant-old", "read", "old output ".repeat(4_000))],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        turn(
            "msg-assistant-recent",
            "assistant",
            [toolItem("msg-assistant-recent", "read", "recent output ".repeat(200))],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "tail user")], 5),
        turn(
            "msg-assistant-tail",
            "assistant",
            [textItem("msg-assistant-tail", "tail assistant")],
            6,
        ),
        turn("msg-user-4", "user", [textItem("msg-user-4", "latest user")], 7),
    ]

    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, force: true, recentToolResultBudgetTokens: 1_500 }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.preservedToolCallIds.length, 1)
    assert.match(plan.preservedToolCallIds[0], /msg-assistant-recent-read-call/)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    const oldAssistantText = syntheticTextOf(
        transformed.find((item) => item.key === "msg-assistant-old"),
    )
    assert.doesNotMatch(oldAssistantText, /old output/)

    const recentAssistant = transformed.find((item) => item.key === "msg-assistant-recent")
    const recentTool = recentAssistant?.items.find((item) => item.kind === "tool")
    assert.equal(recentTool?.kind, "tool")
    if (recentTool?.kind === "tool") {
        assert.match(String((recentTool.handle as TestToolPart).output), /recent output/)
    }
})

test("a single old tool result cannot consume more than the full live target", () => {
    const turns = buildLargeConversation()
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 10_000,
            force: true,
            minTailUserTurns: 1,
            recentToolResultBudgetTokens: 40_000,
            targetTokens: 3_500,
            preservePrefixBudgets: true,
        }),
        spec,
    )
    assert.ok(plan)
    const oversizedCall = turns
        .flatMap((item) => item.items)
        .find((item) => item.kind === "tool" && codec.estimateItem(item) > plan.targetTokens)
    assert.ok(oversizedCall?.kind === "tool")
    assert.ok(!plan.preservedToolCallIds.includes(oversizedCall.callId))
    assert.ok(plan.afterPruneTokens < plan.contextLimit)
})

test("planner preserves only the latest compacted todo state", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                toolItem("msg-assistant-1", "todowrite", "old todos", {
                    todos: [{ content: "obsolete task", status: "pending", priority: "high" }],
                }),
                toolItem("msg-assistant-1", "todowrite", "latest todos", {
                    todos: [{ content: "current task", status: "in_progress", priority: "high" }],
                }),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        turn("msg-assistant-2", "assistant", [textItem("msg-assistant-2", "middle assistant")], 4),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user")], 5),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    const compactedText = syntheticTextOf(
        transformed.find((item) => item.key === "msg-assistant-1"),
    )
    assert.match(compactedText, /Latest todo state preserved/)
    assert.match(compactedText, /current task/)
    assert.doesNotMatch(compactedText, /obsolete task/)
})

test("pruned tools leave compact success and verbatim failure stubs", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                toolItem("msg-assistant-1", "read", "package contents", { filePath: "src/app.ts" }),
                errorToolItem("msg-assistant-1", "bash", "ENOENT: missing config\nstack detail", {
                    command: "npm run test",
                }),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        turn("msg-assistant-2", "assistant", [textItem("msg-assistant-2", "middle assistant")], 4),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user")], 5),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.transcript.content, "")
    assert.deepEqual(plan.transcript.messageIds, ["msg-user-1", "msg-assistant-1"])

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)

    const compacted = transformed.find((item) => item.key === "msg-assistant-1")
    assert.ok(compacted)
    assert.deepEqual(
        compacted.items.map((item) => (item.kind === "synthetic" ? item.text : item.kind)),
        ["[tool:read] src/app.ts — ok", "[tool:bash] npm run test — error: ENOENT: missing config"],
    )
})

test("tool stubs replay byte-stably", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                errorToolItem("msg-assistant-1", "read", "EACCES: denied", {
                    filePath: "src/secret.ts",
                }),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        turn("msg-assistant-2", "assistant", [textItem("msg-assistant-2", "middle assistant")], 4),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user")], 5),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 1_000_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const first = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const replayed = replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true })

    assert.ok(replayed)
    assert.equal(JSON.stringify(replayed), JSON.stringify(first))
    assert.match(JSON.stringify(replayed), /\[tool:read\] src\/secret\.ts — error: EACCES: denied/)
})

test("duplicate tool reads keep only the newest result and stub older failures", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                errorToolItem("msg-assistant-1", "read", "EACCES: denied\nstack", {
                    filePath: "src/./app.ts",
                }),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "next turn")], 3),
        turn(
            "msg-assistant-2",
            "assistant",
            [toolItem("msg-assistant-2", "read", "newest contents", { filePath: "src/app.ts" })],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "middle user")], 5),
        turn("msg-assistant-3", "assistant", [textItem("msg-assistant-3", "middle assistant")], 6),
        turn("msg-user-4", "user", [textItem("msg-user-4", "latest user")], 7),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 1_000_000, force: true, recentToolResultBudgetTokens: 100_000 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const liveTools = transformed
        .flatMap((item) => item.items)
        .filter((item) => item.kind === "tool")
    const older = syntheticTextOf(transformed.find((item) => item.key === "msg-assistant-1"))
    const replayed = replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true })

    assert.equal(liveTools.length, 1)
    assert.match(String((liveTools[0].handle as TestToolPart).output), /newest contents/)
    assert.equal(
        older,
        "[tool:read] src/./app.ts — superseded by later read on src/app.ts; error: EACCES: denied",
    )
    assert.deepEqual(
        plan.stages.slice(0, 4).map((stage) => stage.name),
        ["skills", "supersede-reads", "purge-error-inputs", "tools-old"],
    )
    assert.ok(
        plan.stages.some((stage) => stage.name === "supersede-reads" && stage.status === "applied"),
    )
    assert.ok(replayed)
    assert.equal(JSON.stringify(replayed), JSON.stringify(transformed))
})

test("stale failed tools keep the error but lose their input payload", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                errorToolItem("msg-assistant-1", "bash", "ENOENT: missing config\nstack", {
                    command: "npm run test",
                    secret: "must-not-survive",
                }),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        turn("msg-assistant-2", "assistant", [textItem("msg-assistant-2", "middle assistant")], 4),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user")], 5),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 1_000_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const text = syntheticTextOf(transformed.find((item) => item.key === "msg-assistant-1"))

    assert.equal(text, "[tool:bash] npm run test — error: ENOENT: missing config")
    assert.doesNotMatch(JSON.stringify(transformed), /must-not-survive/)
    assert.ok(
        plan.stages.some(
            (stage) => stage.name === "purge-error-inputs" && stage.status === "applied",
        ),
    )
})

test("purging stale failures preserves the newest failed tool inside the recent window", () => {
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old turn")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                errorToolItem("msg-assistant-1", "read", "ENOENT: old missing", {
                    filePath: "src/old.ts",
                    secret: "old-secret",
                }),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "next turn")], 3),
        turn(
            "msg-assistant-2",
            "assistant",
            [
                errorToolItem("msg-assistant-2", "read", "EACCES: newest denied", {
                    filePath: "src/new.ts",
                    secret: "new-secret",
                }),
            ],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "middle user")], 5),
        turn("msg-assistant-3", "assistant", [textItem("msg-assistant-3", "middle assistant")], 6),
        turn("msg-user-4", "user", [textItem("msg-user-4", "latest user")], 7),
    ]
    const newest = turns[3].items[0]
    assert.equal(newest.kind, "tool")
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 1_000_000,
            force: true,
            recentToolResultBudgetTokens: newest.kind === "tool" ? codec.estimateItem(newest) : 0,
        }),
        spec,
    )
    assert.ok(plan)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const liveTools = transformed
        .flatMap((item) => item.items)
        .filter((item) => item.kind === "tool")
    const serialized = JSON.stringify(transformed)

    assert.deepEqual(plan.preservedToolCallIds, ["msg-assistant-2-read-call"])
    assert.equal(liveTools.length, 1)
    assert.match(serialized, /new-secret/)
    assert.doesNotMatch(serialized, /old-secret/)
    assert.match(serialized, /\[tool:read\] src\/old\.ts — error: ENOENT: old missing/)
})

test("engine prunes on provider-reported usage the raw estimate alone misses", async () => {
    const turns = buildMultiRunConversation()
    const contextLimit = codec.estimateTurns(turns) * 2
    const engine = createEngine(spec, {
        transcripts: {
            citablePath: (key, hash) => `transcripts/${key}/${hash}.md`,
            write: async () => ({}),
        },
        plans: { load: () => null, save: () => {} },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })

    const withoutUsage = await engine.process({ sessionKey, turns, contextLimit })
    assert.equal(withoutUsage.outcome, "unchanged")

    const withUsage = await engine.process({
        sessionKey,
        turns,
        contextLimit,
        providerReportedTokens: Math.floor(contextLimit * 0.9),
    })
    assert.equal(withUsage.outcome, "planned")
})

test("engine declines a complete-context expansion when the target is irreducible", async () => {
    const turns = [
        turn("old-user", "user", [textItem("old-user", "Synthetic violet decision")], 1),
        turn("old-assistant", "assistant", [textItem("old-assistant", "Done")], 2),
        turn("middle-user", "user", [textItem("middle-user", "Keep violet")], 3),
        turn("middle-assistant", "assistant", [textItem("middle-assistant", "Acknowledged")], 4),
        turn("new-user", "user", [textItem("new-user", "Continue the task")], 5),
    ]
    const options = inputs({
        contextLimit: 5_000,
        force: true,
        triggerTokens: 10,
        targetTokens: 50,
        prefixSummaryAllowed: false,
        recentAssistantOutputs: 5,
        preservePrefixBudgets: true,
        summariesAllowed: false,
    })
    const candidate = buildPlan(turns, options, spec)
    assert.ok(candidate)
    assert.ok(candidate.afterPruneTokens >= codec.estimateTurns(turns) + candidate.overheadTokens)
    let saved = false
    let wrote = false
    const engine = createEngine(spec, {
        transcripts: {
            citablePath: (key, hash) => `transcripts/${key}/${hash}.md`,
            write: async () => {
                wrote = true
                return {}
            },
        },
        plans: {
            load: () => null,
            save: () => {
                saved = true
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    const result = await engine.process({ ...options, sessionKey, turns })
    assert.equal(result.outcome, "unchanged")
    assert.equal(saved, false)
    assert.equal(wrote, false)
})

test("engine does not replay thresholds cached for another model", async () => {
    const turns = buildMultiRunConversation()
    const original = buildPlan(
        turns,
        inputs({ contextLimit: 40_000, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(original)
    const snapshot = toPlanSnapshot(original)
    let saved: PlanSnapshot | null | undefined
    const engine = createEngine(spec, {
        transcripts: {
            citablePath: (key, hash) => `transcripts/${key}/${hash}.md`,
            write: async () => ({}),
        },
        plans: {
            load: () => snapshot,
            save: (_key, value) => {
                saved = value
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    const result = await engine.process({
        sessionKey,
        turns,
        contextLimit: 40_000,
        triggerTokens: codec.estimateTurns(turns) + 100_000,
    })
    assert.equal(result.outcome, "unchanged")
    assert.equal(saved, null)
})

test("engine rebuilds a cached plan when prefix-summary or collapse settings change", async () => {
    const turns = buildMultiRunConversation()
    const original = buildPlan(
        turns,
        inputs({
            contextLimit: 40_000,
            recentToolResultBudgetTokens: 0,
            prefixSummaryAllowed: true,
            collapsePercent: 10,
        }),
        spec,
    )
    assert.ok(original)
    let snapshot = toPlanSnapshot(original)
    const engine = createEngine(spec, {
        transcripts: {
            citablePath: (key, hash) => `transcripts/${key}/${hash}.md`,
            write: async () => ({}),
        },
        plans: {
            load: () => snapshot,
            save: (_key, value) => {
                if (value) snapshot = value
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    const request = {
        sessionKey,
        turns,
        contextLimit: 40_000,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
        collapsePercent: 10,
    }
    assert.equal((await engine.process(request)).outcome, "replayed")

    assert.equal((await engine.process({ ...request, collapsePercent: 75 })).outcome, "planned")
    assert.equal(snapshot.collapsePercent, 75)
    assert.equal(
        (await engine.process({ ...request, collapsePercent: 75, prefixSummaryAllowed: false }))
            .outcome,
        "planned",
    )
    assert.equal(snapshot.prefixSummaryAllowed, false)
    const withArchive = {
        ...request,
        collapsePercent: 75,
        prefixSummaryAllowed: false,
        archiveGeneration: 1,
        retirementThrough: 1,
    }
    assert.equal((await engine.process(withArchive)).outcome, "planned")
    assert.equal(snapshot.archiveGeneration, 1)
    assert.equal(snapshot.retirementThrough, 1)
    assert.equal((await engine.process(withArchive)).outcome, "replayed")
    assert.equal(
        (await engine.process({ ...withArchive, archiveGeneration: 2 })).outcome,
        "planned",
    )
})

test("engine keeps the deterministic plan when summary scheduling rejects", async () => {
    const turns = buildMultiRunConversation()
    const planInputs = inputs({ contextLimit: 40_000, recentToolResultBudgetTokens: 0 })
    const expectedPlan = buildPlan(turns, planInputs, spec)
    assert.ok(expectedPlan)
    assert.ok(expectedPlan.summaryJobs.length > 0)
    const warnings: string[] = []
    const warningDetails: unknown[] = []
    let saved: PlanSnapshot | null = null
    const engine = createEngine(spec, {
        transcripts: {
            citablePath: planInputs.citablePath,
            write: async () => ({}),
        },
        plans: {
            load: () => null,
            save: (_key, snapshot) => {
                saved = snapshot
            },
        },
        logger: {
            info() {},
            debug() {},
            warn(message, data) {
                warnings.push(message)
                warningDetails.push(data)
            },
            error() {},
        },
    })

    const result = await engine.process({
        sessionKey,
        turns,
        contextLimit: 40_000,
        recentToolResultBudgetTokens: 0,
        summarize: async () => {
            throw new Error("scheduler failed with PRIVATE_USER_TEXT")
        },
    })

    assert.equal(result.outcome, "planned")
    if (result.outcome !== "planned") return
    assert.equal(
        JSON.stringify(result.turns),
        JSON.stringify(transformTurns(turns, expectedPlan.rawTailStartIndex, expectedPlan, spec)),
    )
    assert.ok(saved)
    assert.ok(warnings.includes("Summary scheduling failed; using deterministic fallback"))
    assert.ok(JSON.stringify(warningDetails).includes("summary_failure"))
    assert.doesNotMatch(JSON.stringify(warningDetails), /PRIVATE_USER_TEXT/)
})

test("verbose keyed turn summaries cannot enlarge a saved plan", async () => {
    const turns = buildMultiRunConversation()
    const planInputs = inputs({
        contextLimit: 40_000,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: false,
    })
    const baseline = buildPlan(turns, planInputs, spec)
    assert.ok(baseline?.summaryJobs.length)
    let snapshot: PlanSnapshot | null = null
    const engine = createEngine(spec, {
        transcripts: { citablePath: planInputs.citablePath, write: async () => ({}) },
        plans: {
            load: () => null,
            save: (_key, plan) => {
                snapshot = plan
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    const result = await engine.process({
        sessionKey,
        turns,
        contextLimit: 40_000,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: false,
        summarize: async (jobs) =>
            Object.fromEntries(
                jobs.map((job) => [
                    job.key,
                    [
                        "## Decisions",
                        `- ${"Rich but verbose historical detail. ".repeat(100)}`,
                        "## Files & Symbols",
                        "- src/file.ts",
                        "## Errors (verbatim)",
                        "- (none)",
                        "## What failed and why",
                        "- (none)",
                        "## Constraints",
                        "- Keep the original requirement.",
                        "## Next step",
                        "- Continue working.",
                    ].join("\n"),
                ]),
            ),
    })
    assert.equal(result.outcome, "planned")
    if (result.outcome !== "planned") return
    assert.ok(result.plan.afterPruneTokens <= baseline.afterPruneTokens)
    assert.deepEqual(result.plan.assistantSummaries, {})
    assert.deepEqual(replayPlanSnapshot(turns, snapshot!, spec), result.turns)
    assert.equal(turns[1].items[1].kind, "text")
})

test("older pruned assistant/tool stubs fold together only when needed and replay exactly", () => {
    const turns = [
        turn("stub-user-1", "user", [textItem("stub-user-1", "Keep the original request.")], 1),
        ...Array.from({ length: 24 }, (_, index) => {
            const key = `stub-assistant-${index}`
            return turn(
                key,
                "assistant",
                [
                    toolItem(key, "bash", `old output ${index} `.repeat(300), {
                        command: `check-${index}`,
                    }),
                ],
                index + 2,
            )
        }),
        turn(
            "stub-user-2",
            "user",
            [textItem("stub-user-2", "Current instruction stays raw.")],
            30,
        ),
        turn("stub-user-3", "user", [textItem("stub-user-3", "Latest correction stays raw.")], 31),
    ]
    const original = JSON.stringify(turns)
    const config = inputs({
        contextLimit: 50_000,
        targetTokens: 500,
        force: true,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: false,
        archiveCatalogText: "- c000001 — Exact previous activity",
    })
    const ungrouped = buildPlan(turns, config, spec)
    assert.ok(ungrouped)
    const options = {
        ...config,
        preservePrefixBudgets: true,
        targetTokens: Math.max(1, ungrouped.afterPruneTokens - 200),
    }
    const plan = buildPlan(turns, options, spec)
    assert.ok(plan)
    assert.ok(plan.stubGroupKeys?.some((keys) => keys.length >= 3))
    assert.ok(plan.afterPruneTokens < ungrouped.afterPruneTokens)
    const stage = plan.stages.find((entry) => entry.name === "assistant-runs")!
    assert.ok(stage.afterTokens < stage.beforeTokens)
    assert.deepEqual(plan.assistantSummaryKeys, [])
    assert.deepEqual(plan.summaryJobs, [])
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.match(applied.map(syntheticTextOf).join("\n"), /Older assistant\/tool activity:/)
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true }),
        applied,
    )
    assert.equal(JSON.stringify(turns), original)
    assert.equal(syntheticTextOf(applied.at(-1)), "Latest correction stays raw.")
    const continued = [
        ...turns,
        turn(
            "stub-next-assistant",
            "assistant",
            [textItem("stub-next-assistant", "Work continues")],
            32,
        ),
        turn("stub-next-user", "user", [textItem("stub-next-user", "Keep the correction")], 33),
    ]
    const next = buildPlan(continued, { ...options, priorPlan: toPlanSnapshot(plan) }, spec)
    assert.ok(next)
    assert.deepEqual(next.stubGroupKeys?.[0], plan.stubGroupKeys?.[0])
    assert.deepEqual(
        replayPlanSnapshot(continued, toPlanSnapshot(next), spec, { allowRegrown: true }),
        transformTurns(continued, next.rawTailStartIndex, next, spec),
    )
})

test("a larger cached per-turn summary cannot be reapplied over a tiny pruned stub", () => {
    const turns = buildMultiRunConversation()
    const options = inputs({
        contextLimit: 40_000,
        force: true,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: false,
    })
    const base = buildPlan(turns, options, spec)
    assert.ok(base)
    const prior = toPlanSnapshot(base)
    prior.assistantSummaries = Object.fromEntries(
        base.assistantSummaryKeys.map((key) => [key, "Old cached narration ".repeat(15_000)]),
    )
    const inflated = buildPlan(
        turns,
        {
            ...options,
            priorPlan: prior,
        },
        spec,
    )
    assert.ok(inflated)
    const stage = inflated.stages.find((entry) => entry.name === "assistant-runs")!
    assert.ok(stage.afterTokens <= stage.beforeTokens)
    assert.ok(
        !transformTurns(turns, inflated.rawTailStartIndex, inflated, spec)
            .map(syntheticTextOf)
            .join("\n")
            .includes("Old cached narration"),
    )
})

test("a last-resort prefix does not schedule invisible assistant-turn summaries", async () => {
    const turns = buildMultiRunConversation()
    const planInputs = inputs({
        contextLimit: 40_000,
        triggerTokens: 500,
        targetTokens: 100,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
    })
    const baseline = buildPlan(turns, planInputs, spec)
    assert.ok(baseline?.requiresCustomCompaction)
    assert.equal(baseline.summaryJobs.length, 0)
    const engine = createEngine(spec, {
        transcripts: { citablePath: planInputs.citablePath, write: async () => ({}) },
        plans: { load: () => null, save: () => {} },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    let calls = 0
    const result = await engine.process({
        sessionKey,
        turns,
        contextLimit: 40_000,
        triggerTokens: 500,
        targetTokens: 100,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
        summarize: async () => {
            calls++
            return {}
        },
    })
    assert.equal(result.outcome, "planned")
    if (result.outcome !== "planned") return
    assert.equal(result.plan.afterPruneTokens, baseline.afterPruneTokens)
    assert.equal(calls, 0)
    assert.deepEqual(result.plan.assistantSummaries, {})
    assert.ok(result.plan.prefixSummary?.includes("First task, keep this requirement."))
})

test("archive handoff is tried after cheap pruning and deterministic prefix is only its fallback", async () => {
    const turns = buildMultiRunConversation()
    const options = inputs({
        contextLimit: 40_000,
        triggerTokens: 500,
        targetTokens: 100,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
        preservePrefixBudgets: true,
        recentAssistantOutputs: 0,
        archiveCatalogText: "- c000001 — Earlier work",
    })
    const cheap = buildPlan(turns, { ...options, prefixSummaryAllowed: false }, spec)
    assert.ok(cheap)
    assert.ok(cheap.afterPruneTokens > 115)
    assert.equal(cheap.requiresCustomCompaction, false)
    const handoff = [
        "## Decisions",
        "- LUNA HANDOFF: work completed.",
        "## Files & Symbols",
        "- src/app.ts",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- (none)",
        "## Constraints",
        "- Keep original intent.",
        "## Next step",
        "- Continue the current work.",
    ].join("\n")
    let saved: PlanSnapshot | null = null
    const engine = createEngine(spec, {
        transcripts: { citablePath: options.citablePath, write: async () => ({}) },
        plans: {
            load: () => null,
            save: (_key, snapshot) => {
                saved = snapshot
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    let calls = 0
    let commits = 0
    const result = await engine.process({
        ...options,
        sessionKey,
        turns,
        summarizeArchive: async (plan) => {
            calls++
            assert.equal(plan.requiresCustomCompaction, false)
            assert.equal(plan.afterPruneTokens, cheap.afterPruneTokens)
            return {
                handoff,
                catalogText: options.archiveCatalogText!,
                commit: async () => {
                    commits++
                },
            }
        },
    })
    assert.equal(calls, 1)
    assert.equal(result.outcome, "planned")
    if (result.outcome !== "planned") return
    assert.equal(commits, 1)
    assert.equal(result.plan.requiresCustomCompaction, true)
    assert.match(result.plan.prefixSummary ?? "", /LUNA HANDOFF/)
    assert.ok(result.plan.afterPruneTokens < cheap.afterPruneTokens)
    assert.ok(
        result.plan.afterPruneTokens > result.plan.targetTokens,
        "a useful handoff need not reach the target",
    )
    assert.ok(
        result.plan.afterPruneTokens < result.plan.triggerTokens,
        "shrinking below the trigger still counts as success",
    )
    assert.deepEqual(replayPlanSnapshot(turns, saved!, spec, { allowRegrown: true }), result.turns)

    const failed = await createEngine(spec, {
        transcripts: { citablePath: options.citablePath, write: async () => ({}) },
        plans: { load: () => null, save: () => {} },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    }).process({
        ...options,
        sessionKey,
        turns,
        summarizeArchive: async (plan) => {
            assert.equal(plan.requiresCustomCompaction, false)
            return null
        },
    })
    assert.equal(failed.outcome, "planned")
    if (failed.outcome === "planned") {
        assert.equal(failed.plan.requiresCustomCompaction, true)
        assert.doesNotMatch(failed.plan.prefixSummary ?? "", /LUNA HANDOFF/)
    }

    const withinBand = await createEngine(spec, {
        transcripts: { citablePath: options.citablePath, write: async () => ({}) },
        plans: { load: () => null, save: () => {} },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    }).process({
        ...options,
        sessionKey,
        turns,
        targetTokens: Math.ceil(cheap.afterPruneTokens / 1.1),
        summarizeArchive: async () => {
            throw new Error("Luna must not run within the band")
        },
    })
    assert.equal(withinBand.outcome, "planned")
    if (withinBand.outcome === "planned")
        assert.equal(withinBand.plan.requiresCustomCompaction, false)
})

test("a saved prefix stays stable while newly covered turns wait native for Luna", () => {
    const original = [
        turn("rollover-user", "user", [textItem("rollover-user", "Keep the old contract")], 1),
        ...Array.from({ length: 8 }, (_, index) =>
            turn(
                `rollover-old-${index}`,
                "assistant",
                [
                    textItem(
                        `rollover-old-${index}`,
                        `Old result ${index}: ${"evidence ".repeat(180)}`,
                    ),
                ],
                index + 2,
            ),
        ),
        turn("rollover-current", "user", [textItem("rollover-current", "Current contract")], 20),
    ]
    const options = inputs({
        contextLimit: 50_000,
        targetTokens: 300,
        force: true,
        minTailUserTurns: 1,
        recentToolResultBudgetTokens: 0,
        preservePrefixBudgets: true,
        archiveCatalogText: "- c000001 — Older work",
    })
    const first = buildPlan(original, options, spec)
    assert.ok(first?.requiresCustomCompaction)
    const continued = [
        ...original,
        turn(
            "rollover-new",
            "assistant",
            [textItem("rollover-new", `NEW NATIVE RESULT: ${"current evidence ".repeat(400)}`)],
            21,
        ),
        turn("rollover-next", "user", [textItem("rollover-next", "New correction stays raw")], 22),
    ]
    const deferred = buildPlan(
        continued,
        {
            ...options,
            priorPlan: toPlanSnapshot(first),
            prefixSummaryAllowed: false,
            deferPrefixConsolidation: true,
        },
        spec,
    )
    assert.ok(deferred?.requiresCustomCompaction)
    assert.equal(deferred.prefixSummary, first.prefixSummary)
    assert.ok(deferred.preservedPrefixTurnKeys?.includes("rollover-new"))
    assert.doesNotMatch(deferred.prefixSummary ?? "", /NEW NATIVE RESULT/)
    const applied = transformTurns(continued, deferred.rawTailStartIndex, deferred, spec)
    assert.match(applied.map(syntheticTextOf).join("\n"), /NEW NATIVE RESULT/)
    assert.deepEqual(
        replayPlanSnapshot(continued, toPlanSnapshot(deferred), spec, { allowRegrown: true }),
        applied,
    )
})

test("a bounded prefix synthesis replaces the fallback once and replays identically", async () => {
    const turns = buildMultiRunConversation()
    const options = inputs({
        contextLimit: 40_000,
        triggerTokens: 500,
        targetTokens: 100,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
    })
    const baseline = buildPlan(turns, options, spec)
    assert.ok(baseline?.requiresCustomCompaction)
    const summary = [
        "## Decisions",
        "- Completed work in src/app.ts with the current task intact.",
        "## Files & Symbols",
        "- src/app.ts",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- (none)",
        "## Constraints",
        "- First task, keep this requirement.",
        "## Next step",
        "- Continue the latest request.",
    ].join("\n")
    let saved: PlanSnapshot | null = null
    let calls = 0
    const ports = {
        transcripts: { citablePath: options.citablePath, write: async () => ({}) },
        plans: {
            load: () => saved,
            save: (_key: string, snapshot: PlanSnapshot | null) => {
                saved = snapshot
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    }
    const engine = createEngine(spec, ports)
    const first = await engine.process({
        sessionKey,
        turns,
        contextLimit: 40_000,
        triggerTokens: 500,
        targetTokens: 100,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
        summarizePrefix: async () => {
            calls++
            return summary
        },
    })
    assert.equal(first.outcome, "planned")
    if (first.outcome !== "planned") return
    assert.equal(calls, 1)
    assert.ok(first.plan.afterPruneTokens < baseline.afterPruneTokens)
    assert.equal(first.plan.prefixSummary, summary)
    assert.deepEqual(first.plan.summaryJobs, [])
    const replay = await engine.process({
        sessionKey,
        turns,
        contextLimit: 40_000,
        triggerTokens: 500,
        targetTokens: 100,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: true,
        summarizePrefix: async () => {
            calls++
            return summary
        },
    })
    assert.equal(replay.outcome, "replayed")
    if (replay.outcome === "replayed") assert.deepEqual(replay.turns, first.turns)
    assert.equal(calls, 1)
})

test("a consolidated prefix absorbs old per-turn summaries instead of caching them again", () => {
    const turns = buildMultiRunConversation()
    const common = { force: true, recentToolResultBudgetTokens: 0, prefixSummaryAllowed: true }
    const selected = buildPlan(turns, inputs({ ...common, contextLimit: 9_000 }), spec)
    assert.ok(selected?.summaryJobs.length)
    const key = selected.summaryJobs[0].key
    const summary = "Completed the distinctive earlier task state in src/earlier.ts."
    const prefixOptions = { ...common, contextLimit: 40_000, triggerTokens: 1, targetTokens: 1 }
    const first = buildPlan(
        turns,
        inputs({ ...prefixOptions, assistantSummaries: { [key]: summary } }),
        spec,
    )
    assert.ok(first?.requiresCustomCompaction)
    assert.match(first.prefixSummary ?? "", /distinctive earlier task state/)
    assert.deepEqual(first.assistantSummaryKeys, [])
    assert.deepEqual(first.assistantSummaries, {})
    const snapshot = toPlanSnapshot(first)
    assert.deepEqual(snapshot.assistantSummaryKeys, [])
    assert.deepEqual(snapshot.assistantSummaries, {})

    // An old persisted snapshot may still contain hundreds of absorbed
    // summaries. They are ignored during the next rebuild, not concatenated
    // into the intermediate context a second time.
    const legacy = {
        ...snapshot,
        assistantSummaryKeys: [key],
        assistantSummaries: { [key]: summary.repeat(100) },
    }
    const clean = buildPlan(turns, inputs({ ...prefixOptions, priorPlan: snapshot }), spec)
    const migrated = buildPlan(turns, inputs({ ...prefixOptions, priorPlan: legacy }), spec)
    assert.ok(clean?.requiresCustomCompaction && migrated?.requiresCustomCompaction)
    assert.equal(migrated.prefixSummary, clean.prefixSummary)
    assert.equal(migrated.afterPruneTokens, clean.afterPruneTokens)
    assert.deepEqual(migrated.assistantSummaryKeys, [])
    assert.deepEqual(migrated.assistantSummaries, {})
    const replay = replayPlanSnapshot(turns, toPlanSnapshot(migrated), spec, {
        allowRegrown: true,
    })
    assert.deepEqual(replay, transformTurns(turns, migrated.rawTailStartIndex, migrated, spec))

    const grown = [
        ...turns,
        turn("msg-assistant-new", "assistant", [textItem("msg-assistant-new", "New work")], 8),
        turn("msg-user-new", "user", [textItem("msg-user-new", "Continue the task")], 9),
    ]
    const extended = buildPlan(
        grown,
        inputs({ ...prefixOptions, priorPlan: toPlanSnapshot(migrated) }),
        spec,
    )
    assert.ok(extended?.requiresCustomCompaction)
    assert.match(extended.prefixSummary ?? "", /distinctive earlier task state/)
    assert.deepEqual(extended.assistantSummaries, {})
})

test("planner triggers when either the provider total or the raw estimate crosses the trigger", () => {
    const turns = buildLargeConversation()
    const estimate = codec.estimateTurns(turns)
    const contextLimit = Math.max(1, Math.floor(estimate / 0.9))
    assert.ok(estimate > Math.floor(contextLimit * 0.85))

    // Provider total lags behind fresh turns the estimate already sees.
    const plan = buildPlan(turns, inputs({ contextLimit, providerReportedTokens: 10 }), spec)
    assert.ok(plan)

    // Neither scale over the trigger: no plan.
    const calm = buildPlan(
        turns,
        inputs({ contextLimit: estimate * 4, providerReportedTokens: 10 }),
        spec,
    )
    assert.equal(calm, null)
})

test("replacement plans keep prior pruning stages and preserved tools as a monotonic floor", () => {
    const turns = buildLargeConversation()
    const first = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(first)
    assert.ok(
        first.stages.some((stage) => stage.name === "reasoning" && stage.status !== "skipped"),
    )

    const replacement = buildPlan(
        turns,
        inputs({
            contextLimit: 1_000_000,
            force: true,
            recentToolResultBudgetTokens: 80_000,
            priorPlan: toPlanSnapshot(first),
        }),
        spec,
    )

    assert.ok(replacement)
    // A generous limit alone would skip reasoning; the prior plan already
    // pruned it, so the replacement must not resurrect it.
    assert.ok(
        replacement.stages.some(
            (stage) => stage.name === "reasoning" && stage.status !== "skipped",
        ),
    )
    // Tool results the model already lost inside the prior compacted prefix
    // stay lost even under a larger recent-tool budget.
    const priorPreserved = new Set(first.preservedToolCallIds)
    for (const callId of replacement.preservedToolCallIds) {
        if (priorPreserved.has(callId)) continue
        const previouslyCompacted = turns
            .slice(0, first.rawTailStartIndex)
            .some((item) =>
                item.items.some((part) => part.kind === "tool" && part.callId === callId),
            )
        assert.equal(previouslyCompacted, false)
    }
})

test("replacement plans reuse prior assistant summaries without new summary jobs", () => {
    const turns = buildMultiRunConversation()
    const first = buildPlan(
        turns,
        inputs({ contextLimit: 9_000, recentToolResultBudgetTokens: 0, force: true }),
        spec,
    )
    assert.ok(first)
    assert.ok(first.summaryJobs.length > 0)
    const summaries = Object.fromEntries(
        first.summaryJobs.map((job) => [job.key, "Accepted summary."]),
    )
    const settled = buildPlan(
        turns,
        inputs({
            contextLimit: 9_000,
            recentToolResultBudgetTokens: 0,
            force: true,
            assistantSummaries: summaries,
        }),
        spec,
    )
    assert.ok(settled)
    assert.equal(settled.summaryJobs.length, 0)

    const replacement = buildPlan(
        turns,
        inputs({
            contextLimit: 9_000,
            recentToolResultBudgetTokens: 0,
            force: true,
            priorPlan: toPlanSnapshot(settled),
        }),
        spec,
    )

    assert.ok(replacement)
    assert.equal(replacement.summaryJobs.length, 0)
    assert.deepEqual(replacement.assistantSummaries, settled.assistantSummaries)
})

test("expanded prefix summaries include newly compacted user context", () => {
    const firstTurns = [
        turn("u1", "user", [textItem("u1", "old user")], 1),
        turn("a1", "assistant", [textItem("a1", "old detail ".repeat(2_000))], 2),
        turn("u2", "user", [textItem("u2", "middle user")], 3),
        turn("a2", "assistant", [textItem("a2", "middle detail ".repeat(2_000))], 4),
        turn("u3", "user", [textItem("u3", "newly crossed requirement")], 5),
    ]
    const first = buildPlan(
        firstTurns,
        inputs({ contextLimit: 100, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(first?.requiresCustomCompaction)
    const grown = [
        ...firstTurns,
        turn("a3", "assistant", [textItem("a3", "later detail ".repeat(2_000))], 6),
        turn("u4", "user", [textItem("u4", "later user")], 7),
        turn("a4", "assistant", [textItem("a4", "latest assistant")], 8),
        turn("u5", "user", [textItem("u5", "latest user")], 9),
    ]

    const prior = toPlanSnapshot(first)
    prior.prefixSummary = "PRIOR CHECKPOINT ONLY"
    const replacement = buildPlan(
        grown,
        inputs({
            contextLimit: 100,
            force: true,
            recentToolResultBudgetTokens: 0,
            priorPlan: prior,
        }),
        spec,
    )

    assert.ok(replacement?.requiresCustomCompaction)
    assert.match(replacement.prefixSummary ?? "", /newly crossed requirement/)
    assert.doesNotMatch(replacement.prefixSummary ?? "", /PRIOR CHECKPOINT ONLY/)

    const repeated = buildPlan(
        grown,
        inputs({
            contextLimit: 100,
            force: true,
            recentToolResultBudgetTokens: 0,
            priorPlan: prior,
        }),
        spec,
    )
    assert.ok(repeated)
    assert.equal(repeated.prefixSummary, replacement.prefixSummary)
    assert.equal(
        JSON.stringify(transformTurns(grown, repeated.rawTailStartIndex, repeated, spec)),
        JSON.stringify(transformTurns(grown, replacement.rawTailStartIndex, replacement, spec)),
    )
})

test("available summarizer rolls a prior prefix summary over only the newly compacted delta", async () => {
    const firstTurns = [
        turn("u1", "user", [textItem("u1", "old user")], 1),
        turn("a1", "assistant", [textItem("a1", "old detail ".repeat(2_000))], 2),
        turn("u2", "user", [textItem("u2", "middle user")], 3),
        turn("a2", "assistant", [textItem("a2", "middle detail ".repeat(2_000))], 4),
        turn("u3", "user", [textItem("u3", "newly crossed requirement")], 5),
    ]
    const first = buildPlan(
        firstTurns,
        inputs({ contextLimit: 100, force: true, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(first?.requiresCustomCompaction)

    const priorSummary = [
        "## Decisions",
        "- PRIOR CHECKPOINT DECISION",
        "",
        "## Files & Symbols",
        "- src/prior.ts",
        "",
        "## Errors (verbatim)",
        "- (none)",
        "",
        "## What failed and why",
        "- (none)",
        "",
        "## Constraints",
        "- Preserve the prior checkpoint.",
        "",
        "## Next step",
        "- Continue from the delta.",
    ].join("\n")
    const rolledSummary = priorSummary.replace(
        "- PRIOR CHECKPOINT DECISION",
        "- ROLLED CHECKPOINT WITH NEW DELTA",
    )
    let stored: PlanSnapshot = { ...toPlanSnapshot(first), prefixSummary: priorSummary }
    const capturedJobs: Array<{
        key: string
        prompt: string
        rangeStartMessageId: string
        rangeEndMessageId: string
    }> = []
    const engine = createEngine(spec, {
        transcripts: {
            citablePath: (key, hash) => `transcripts/${key}/${hash}.md`,
            write: async () => ({}),
        },
        plans: {
            load: () => stored,
            save: (_key, snapshot) => {
                if (snapshot) stored = snapshot
            },
        },
        logger: { info() {}, debug() {}, warn() {}, error() {} },
    })
    const grown = [
        ...firstTurns,
        turn("a3", "assistant", [textItem("a3", "later detail ".repeat(2_000))], 6),
        turn("u4", "user", [textItem("u4", "still raw later user")], 7),
        turn("a4", "assistant", [textItem("a4", "still raw latest assistant")], 8),
        turn("u5", "user", [textItem("u5", "still raw latest user")], 9),
    ]

    const result = await engine.process({
        sessionKey,
        turns: grown,
        contextLimit: 100,
        recentToolResultBudgetTokens: 0,
        summarize: async (jobs) => {
            capturedJobs.push(...jobs)
            return Object.fromEntries(
                jobs.map((job) => [
                    job.key,
                    job.key.startsWith("prefix-summary:") ? rolledSummary : priorSummary,
                ]),
            )
        },
    })

    assert.equal(result.outcome, "planned")
    if (result.outcome !== "planned") return
    const rollingJobs = capturedJobs.filter((job) => job.key.startsWith("prefix-summary:"))
    assert.equal(rollingJobs.length, 1)
    assert.equal(rollingJobs[0].rangeStartMessageId, "u2")
    assert.equal(rollingJobs[0].rangeEndMessageId, "a3")
    assert.match(rollingJobs[0].prompt, /PRIOR CHECKPOINT DECISION/)
    assert.match(rollingJobs[0].prompt, /newly crossed requirement/)
    for (const header of [
        "## Decisions",
        "## Files & Symbols",
        "## Errors (verbatim)",
        "## What failed and why",
        "## Constraints",
        "## Next step",
    ]) {
        assert.match(rollingJobs[0].prompt, new RegExp(header.replace(/[()]/g, "\\$&")))
    }
    assert.match(
        rollingJobs[0].prompt,
        /Preserve exact paths, symbols, error strings, and IDs verbatim/,
    )
    assert.doesNotMatch(rollingJobs[0].prompt, /still raw later user/)
    assert.doesNotMatch(rollingJobs[0].prompt, /still raw latest user/)
    assert.equal(result.plan.prefixSummary, rolledSummary)
    assert.equal(result.plan.requiresCustomCompaction, true)
    assert.equal(result.plan.summaryJobs.length, 0)
    assert.equal(Object.hasOwn(result.plan.assistantSummaries, rollingJobs[0].key), false)
})

test("custom compaction stays sticky for replacement plans", () => {
    const turns = buildMultiRunConversation()
    const first = buildPlan(
        turns,
        inputs({ contextLimit: 500, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(first?.requiresCustomCompaction)

    // A huge limit would normally clear custom compaction entirely, but the
    // model already saw a summarized prefix; reverting would resurrect it.
    const replacement = buildPlan(
        turns,
        inputs({ contextLimit: 1_000_000, force: true, priorPlan: toPlanSnapshot(first) }),
        spec,
    )

    assert.ok(replacement)
    assert.equal(replacement.requiresCustomCompaction, true)
    assert.equal(replacement.prefixSummary, first.prefixSummary)
})

test("ephemeral turns do not count as protected user turns", () => {
    const ignored = turn(
        "msg-ignored",
        "user",
        [textItem("msg-ignored", "Better Compact report")],
        4,
    )
    ignored.ephemeral = true
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", "old user")], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [textItem("msg-assistant-1", "old detail ".repeat(2_000))],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "middle user")], 3),
        ignored,
        turn("msg-assistant-2", "assistant", [textItem("msg-assistant-2", "middle assistant")], 5),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest user")], 6),
    ]

    const plan = buildPlan(turns, inputs({ contextLimit: 10_000, force: true }), spec)

    assert.ok(plan)
    assert.equal(plan.rawTailStartMessageId, "msg-user-2")
})

test("assistant summaries survive a fork that mints new turn keys", () => {
    const turns = buildMultiRunConversation()
    const first = buildPlan(
        turns,
        inputs({ contextLimit: 9_000, recentToolResultBudgetTokens: 0, force: true }),
        spec,
    )
    assert.ok(first)
    assert.ok(first.summaryJobs.length > 0)
    const summaries = Object.fromEntries(
        first.summaryJobs.map((job) => [job.key, "Accepted summary."]),
    )

    const forked = buildMultiRunConversation().map((item) => ({ ...item, key: `fork-${item.key}` }))
    const replay = buildPlan(
        forked,
        inputs({
            contextLimit: 9_000,
            recentToolResultBudgetTokens: 0,
            force: true,
            assistantSummaries: summaries,
        }),
        spec,
    )

    assert.ok(replay)
    assert.equal(replay.summaryJobs.length, 0)
    assert.deepEqual(Object.keys(replay.assistantSummaries).sort(), Object.keys(summaries).sort())
})

test("an oversized turn prunes older items while keeping its newest item raw", () => {
    const sourceTool = toolItem("msg-assistant-large", "read", "old output ".repeat(4_000))
    const newest = textItem("msg-assistant-large-newest", "newest assistant detail stays raw")
    const source = turn("msg-assistant-large", "assistant", [sourceTool, newest], 1)
    const turns = [
        source,
        turn("msg-user-new", "user", [textItem("msg-user-new", "latest request")], 2),
    ]

    const plan = buildPlan(turns, inputs({ contextLimit: 10_000 }), spec)

    assert.ok(plan)
    assert.equal(plan.rawTailStartMessageId, source.key)
    assert.deepEqual(plan.rawTailItemBoundary, { itemKey: newest.key, side: "before" })
    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const transformedSource = transformed.find((candidate) => candidate.handle === source.handle)
    assert.ok(transformedSource)
    assert.equal(transformed.filter((candidate) => candidate.handle === source.handle).length, 1)
    assert.equal(transformedSource.items.at(-1), newest)
    assert.ok(
        transformedSource.items.some(
            (item) => item.kind === "synthetic" && item.text.startsWith("[tool:read]"),
        ),
    )

    const replayed = replayPlanSnapshot(turns, toPlanSnapshot(plan), spec)
    assert.ok(replayed)
    assert.equal(JSON.stringify(replayed), JSON.stringify(transformed))

    const missingBoundary = turns.map((candidate) =>
        candidate === source
            ? { ...candidate, items: candidate.items.filter((item) => item !== newest) }
            : candidate,
    )
    assert.equal(replayPlanSnapshot(missingBoundary, toPlanSnapshot(plan), spec), null)

    const editedPrefix = [
        {
            ...source,
            items: [{ ...sourceTool, key: `${sourceTool.key}-edited` }, newest],
        },
        turns[1],
    ]
    assert.equal(replayPlanSnapshot(editedPrefix, toPlanSnapshot(plan), spec), null)
})

test("an oversized assistant text prefix is summarized without rewriting its raw suffix", () => {
    const oldText = textItem("msg-assistant-text-old", "old assistant detail ".repeat(3_000))
    const newest = textItem("msg-assistant-text-new", "newest assistant conclusion")
    const source = turn("msg-assistant-text", "assistant", [oldText, newest], 1)
    const turns = [
        source,
        turn("msg-user-new", "user", [textItem("msg-user-new", "latest request")], 2),
    ]

    const plan = buildPlan(turns, inputs({ contextLimit: 10_000 }), spec)

    assert.ok(plan)
    assert.deepEqual(plan.rawTailItemBoundary, { itemKey: newest.key, side: "before" })
    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const transformedSource = transformed.find((candidate) => candidate.handle === source.handle)
    assert.ok(transformedSource)
    assert.equal(transformedSource.items.at(-1), newest)
    assert.ok(
        transformedSource.items.some(
            (item) => item.kind === "synthetic" && item.text.startsWith("[Assistant turn summary]"),
        ),
    )
    assert.equal(
        transformedSource.items.some((item) => item === oldText),
        false,
    )
})

test("advancing a split boundary does not reuse a partial assistant summary", () => {
    const oldest = textItem("msg-assistant-oldest", "oldest detail ".repeat(3_000))
    const middle = textItem("msg-assistant-middle", "middle detail ".repeat(3_000))
    const newest = textItem("msg-assistant-newest", "newest detail stays raw")
    const source = turn("msg-assistant-growing", "assistant", [oldest, middle, newest], 1)
    const turns = [
        source,
        turn("msg-user-new", "user", [textItem("msg-user-new", "latest request")], 2),
    ]
    const first = buildPlan(
        turns,
        inputs({ contextLimit: 40_000, triggerRatio: 0.4, targetRatio: 0.3 }),
        spec,
    )
    assert.ok(first)
    assert.deepEqual(first.rawTailItemBoundary, { itemKey: middle.key, side: "before" })
    assert.equal(first.summaryJobs.length, 1)
    const firstSummaryKey = first.summaryJobs[0].key
    const prior = toPlanSnapshot(first)
    prior.assistantSummaries = { [firstSummaryKey]: "accepted partial summary" }
    prior.assistantSummaryKeys = [firstSummaryKey]

    const replacement = buildPlan(
        turns,
        inputs({
            contextLimit: 40_000,
            triggerRatio: 0.4,
            targetRatio: 0.1,
            priorPlan: prior,
        }),
        spec,
    )

    assert.ok(replacement)
    assert.deepEqual(replacement.rawTailItemBoundary, {
        itemKey: newest.key,
        side: "before",
    })
    assert.notEqual(replacement.rangeHash, first.rangeHash)
    assert.equal(replacement.summaryJobs.length, 1)
    assert.notEqual(replacement.summaryJobs[0].key, firstSummaryKey)
})

test("a lone giant tool item is stubbed atomically instead of split", () => {
    const giantTool = toolItem("msg-assistant-tool", "read", "giant output ".repeat(5_000))
    const source = turn("msg-assistant-tool", "assistant", [giantTool], 2)
    const turns = [
        turn("msg-user-old", "user", [textItem("msg-user-old", "old request")], 1),
        source,
        turn("msg-user-new", "user", [textItem("msg-user-new", "latest request")], 3),
    ]

    const plan = buildPlan(turns, inputs({ contextLimit: 10_000 }), spec)

    assert.ok(plan)
    assert.equal(plan.rawTailStartMessageId, source.key)
    assert.deepEqual(plan.rawTailItemBoundary, { itemKey: giantTool.key, side: "after" })
    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const transformedSource = transformed.find((candidate) => candidate.handle === source.handle)
    assert.ok(transformedSource)
    assert.equal(transformed.filter((candidate) => candidate.handle === source.handle).length, 1)
    assert.equal(transformedSource.items.filter((item) => item.kind === "tool").length, 0)
    assert.equal(
        transformedSource.items.filter(
            (item) => item.kind === "synthetic" && item.text.startsWith("[tool:read]"),
        ).length,
        1,
    )

    const replayed = replayPlanSnapshot(turns, toPlanSnapshot(plan), spec)
    assert.ok(replayed)
    assert.equal(JSON.stringify(replayed), JSON.stringify(transformed))

    const appended = textItem("msg-assistant-tool-new", "new same-turn detail stays raw")
    const grownSource = { ...source, items: [giantTool, appended] }
    const grown = [turns[0], grownSource, turns[2]]
    const replayedGrown = replayPlanSnapshot(grown, toPlanSnapshot(plan), spec)
    assert.ok(replayedGrown)
    const replayedGrownSource = replayedGrown.find(
        (candidate) => candidate.handle === source.handle,
    )
    assert.ok(replayedGrownSource)
    assert.equal(replayedGrownSource.items.at(-1), appended)

    const replacement = buildPlan(
        grown,
        inputs({ contextLimit: 10_000, priorPlan: toPlanSnapshot(plan) }),
        spec,
    )
    assert.ok(replacement)
    assert.deepEqual(replacement.rawTailItemBoundary, {
        itemKey: appended.key,
        side: "before",
    })
    const replaced = transformTurns(grown, replacement.rawTailStartIndex, replacement, spec)
    const replacedSource = replaced.find((candidate) => candidate.handle === source.handle)
    assert.ok(replacedSource)
    assert.equal(replacedSource.items.at(-1), appended)
})

test("a protected turn below the trigger stays whole even when it exceeds the target", () => {
    const turns = [
        turn("msg-user-old", "user", [textItem("msg-user-old", "old request")], 1),
        turn(
            "msg-assistant-old",
            "assistant",
            [toolItem("msg-assistant-old", "read", "old output ".repeat(5_000))],
            2,
        ),
        turn("msg-user-middle", "user", [textItem("msg-user-middle", "middle request")], 3),
        turn(
            "msg-assistant-tail",
            "assistant",
            [
                textItem("msg-assistant-tail-old", "tail detail ".repeat(1_000)),
                textItem("msg-assistant-tail-new", "newest tail detail"),
            ],
            4,
        ),
        turn("msg-user-new", "user", [textItem("msg-user-new", "latest request")], 5),
    ]

    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 10_000, recentToolResultBudgetTokens: 0 }),
        spec,
    )

    assert.ok(plan)
    assert.equal(plan.rawTailStartMessageId, "msg-user-middle")
    assert.equal(plan.rawTailItemBoundary, undefined)
})

test("prefix summary keeps user prose verbatim and truncates assistant narration", () => {
    const userText = "the constraint: never split a user turn\nsecond line: ship it\nthird: done"
    const assistantText = "assistant narration ".repeat(2_000)
    const turns = [
        turn("msg-user-1", "user", [textItem("msg-user-1", userText)], 1),
        turn(
            "msg-assistant-1",
            "assistant",
            [
                reasoningItem("msg-assistant-1", "r".repeat(2_000)),
                textItem("msg-assistant-1", assistantText),
                toolItem("msg-assistant-1", "read", "o".repeat(2_000)),
            ],
            2,
        ),
        turn("msg-user-2", "user", [textItem("msg-user-2", "and another thing")], 3),
        turn(
            "msg-assistant-tail",
            "assistant",
            [textItem("msg-assistant-tail", "tail stays " + "x".repeat(4_000))],
            4,
        ),
        turn("msg-user-3", "user", [textItem("msg-user-3", "latest " + "u".repeat(4_000))], 5),
    ]
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 500, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.requiresCustomCompaction, true)
    assert.ok(plan.stages.some((stage) => stage.name === "prefix-summary"))

    const summary = plan.prefixSummary
    assert.ok(summary)
    // User turns are the contract the session answers to: byte-for-byte,
    // newlines included — no bullet rewrapping, no truncation.
    assert.ok(summary.includes(userText))
    // Assistant narration is previewed, not carried whole: a 40KB prose blob
    // lands as a bounded "resume" line.
    assert.ok(!summary.includes(assistantText))
    assert.match(summary, /Resume from prior assistant progress:/)
    assert.match(summary, /\[\.{3}omitted\]/)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.equal(transformed.at(-1)?.key, "msg-user-3")
})

test("summariesAllowed false collapses assistant runs without queuing summary jobs", () => {
    const turns = buildMultiRunConversation()
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 40_000,
            recentToolResultBudgetTokens: 0,
            summariesAllowed: false,
        }),
        spec,
    )
    assert.ok(plan)
    // The stage still applies — the collapse is deterministic — but nothing
    // is queued for the side model.
    assert.ok(
        plan.stages.some((stage) => stage.name === "assistant-runs" && stage.status === "applied"),
    )
    assert.equal(plan.summaryJobs.length, 0)

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const collapsed = transformed.find((item) => item.key === "msg-assistant-big")
    assert.ok(collapsed)
    assert.ok(collapsed.items.every((item) => item.kind === "synthetic"))
    const text = syntheticTextOf(collapsed)
    // The deterministic preview plus the transcript pointer stand in for the
    // summary body.
    assert.match(text, /\[Assistant turn summary\]/)
    assert.match(text, /Raw transcript:/)
})

test("prefixSummaryAllowed false leaves the prefix un-merged even when the target is unreachable", () => {
    const turns = buildMultiRunConversation()
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 500,
            recentToolResultBudgetTokens: 0,
            prefixSummaryAllowed: false,
        }),
        spec,
    )
    assert.ok(plan)
    // The last resort is off: no prefix-summary stage, no custom compaction
    // request — the caller chooses between a rewrite-only answer and a
    // decline instead of a merged prefix it did not opt into.
    assert.equal(plan.requiresCustomCompaction, false)
    assert.ok(!plan.stages.some((stage) => stage.name === "prefix-summary"))

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.ok(!transformed[0]?.key.startsWith("better_compact_summary_"))
})

test("prefixSummaryAllowed with the last resort preserves the prior merge behavior", () => {
    const turns = buildMultiRunConversation()
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 500,
            recentToolResultBudgetTokens: 0,
            prefixSummaryAllowed: true,
        }),
        spec,
    )
    assert.ok(plan)
    assert.equal(plan.requiresCustomCompaction, true)
    assert.ok(plan.stages.some((stage) => stage.name === "prefix-summary"))

    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.ok(transformed[0]?.key.startsWith("better_compact_summary_"))
})

test("summary prompts carry a run's reasoning even though the reasoning stage strips it first", () => {
    // Reasoning is stripped before runs are keyed so plan and replay hash the
    // same items; the summarizer must still see it, or the causal record of
    // why the assistant did what it did is deleted rather than distilled.
    const turns = buildMultiRunConversation()
    const plan = buildPlan(
        turns,
        inputs({ contextLimit: 40_000, recentToolResultBudgetTokens: 0 }),
        spec,
    )
    assert.ok(plan)
    const job = plan.summaryJobs.find(
        (candidate) => candidate.rangeStartMessageId === "msg-assistant-big",
    )
    assert.ok(job, "the big run must be selected for a summary")
    assert.match(job.prompt, /big private reasoning/)

    // The working prefix the plan committed has no reasoning left in it.
    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    assert.ok(transformed.every((item) => item.items.every((part) => part.kind !== "reasoning")))
})

test("recent whole reasoning parts scale with history and replay without resurrecting old parts", () => {
    const reasoning = "Working through a still-relevant implementation detail. ".repeat(20)
    const itemCost = countTokens(codec.transcriptLine(reasoningItem("sample", reasoning)))
    const conversation = (count: number): Turn[] => {
        const turns: Turn[] = []
        for (let index = 0; index < count + 2; index++) {
            turns.push(
                turn(
                    `user-${index}`,
                    "user",
                    [textItem(`user-${index}`, `instruction ${index}`)],
                    index * 2 + 1,
                ),
            )
            turns.push(
                turn(
                    `assistant-${index}`,
                    "assistant",
                    [
                        reasoningItem(`assistant-${index}`, reasoning),
                        textItem(`assistant-${index}`, `result ${index}`),
                    ],
                    index * 2 + 2,
                ),
            )
        }
        return turns
    }
    const reasoningOnly: LadderSpec = { ...spec, stages: [{ ...reasoningStage, always: true }] }
    const make = (count: number, priorPlan?: PlanSnapshot, base = Math.floor(itemCost / 2)) => {
        const turns = conversation(count)
        const plan = buildPlan(
            turns,
            inputs({
                contextLimit: 100_000,
                triggerTokens: 1,
                targetTokens: 20_000,
                force: true,
                prefixSummaryAllowed: false,
                recentReasoningBudgetTokens: base,
                priorPlan,
            }),
            reasoningOnly,
        )
        assert.ok(plan)
        return { plan, turns }
    }
    const small = make(4)
    const large = make(8)
    assert.equal(small.plan.preservedReasoningItemKeys?.length, 1)
    assert.equal(large.plan.preservedReasoningItemKeys?.length, 2)
    const applied = transformTurns(
        large.turns,
        large.plan.rawTailStartIndex,
        large.plan,
        reasoningOnly,
    )
    const replayed = replayPlanSnapshot(large.turns, toPlanSnapshot(large.plan), reasoningOnly, {
        allowRegrown: true,
    })
    assert.deepEqual(replayed, applied)
    const stripped = make(4, undefined, 0)
    const grown = make(4, toPlanSnapshot(stripped.plan), itemCost * 4)
    assert.equal(
        grown.plan.preservedReasoningItemKeys?.length,
        0,
        "already-pruned reasoning cannot reappear",
    )
    const tinyWindow = buildPlan(
        conversation(4),
        inputs({
            contextLimit: 1_000,
            triggerTokens: 1,
            targetTokens: 1,
            force: true,
            prefixSummaryAllowed: false,
            recentReasoningBudgetTokens: 20_000,
        }),
        reasoningOnly,
    )
    assert.ok(tinyWindow)
    assert.equal(
        tinyWindow.preservedReasoningItemKeys?.length,
        tinyWindow.transcript.turns
            ?.flatMap((turn) => turn.items)
            .filter((item) => item.kind === "reasoning").length,
        "model context does not reduce the base allowance",
    )
    const noHeadroom = buildPlan(
        conversation(8),
        inputs({
            contextLimit: 100_000,
            triggerTokens: 1,
            targetTokens: 1,
            force: true,
            prefixSummaryAllowed: false,
            recentReasoningBudgetTokens: Math.floor(itemCost / 2),
        }),
        reasoningOnly,
    )
    assert.ok(noHeadroom)
    assert.equal(noHeadroom.preservedReasoningItemKeys?.length, 0, "growth needs target headroom")
})

test("last-resort prefix emits the selected recent reasoning parts and replays them exactly", () => {
    const reasoning = "A significant implementation decision with supporting detail. ".repeat(25)
    const itemCost = countTokens(codec.transcriptLine(reasoningItem("sample", reasoning)))
    const turns: Turn[] = []
    for (let index = 0; index < 7; index++) {
        turns.push(
            turn(
                `reason-user-${index}`,
                "user",
                [textItem(`reason-user-${index}`, `Request ${index}`)],
                index * 2 + 1,
            ),
        )
        turns.push(
            turn(
                `reason-assistant-${index}`,
                "assistant",
                [
                    reasoningItem(`reason-assistant-${index}`, `${reasoning} ${index}`),
                    textItem(`reason-assistant-${index}`, `Decision ${index}`),
                ],
                index * 2 + 2,
            ),
        )
    }
    const reasoningFirst: LadderSpec = { ...spec, stages: [{ ...reasoningStage, always: true }] }
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 100_000,
            triggerTokens: 1,
            targetTokens: 300,
            force: true,
            recentReasoningBudgetTokens: itemCost * 2 + 20,
            prefixSummaryAllowed: true,
            preservePrefixBudgets: true,
            prefixSummary:
                "## Decisions\n- Preserve current state.\n## Next step\n- Continue work.",
        }),
        reasoningFirst,
    )
    assert.ok(plan?.requiresCustomCompaction)
    assert.equal(plan.reasoningSurvivesPrefix, true)
    const selected = new Set(plan.preservedReasoningItemKeys)
    assert.ok(selected.size >= 1)
    const applied = transformTurns(turns, plan.rawTailStartIndex, plan, reasoningFirst)
    const emitted = applied
        .flatMap((entry) => entry.items)
        .filter((item) => item.kind === "reasoning" && selected.has(item.key))
    assert.equal(
        emitted.length,
        selected.size,
        "every selected older reasoning part survives the final handoff",
    )
    assert.equal(plan.afterPruneTokens, codec.estimateTurns(applied) + plan.overheadTokens)
    const snapshot = toPlanSnapshot(plan)
    assert.deepEqual(
        replayPlanSnapshot(turns, snapshot, reasoningFirst, { allowRegrown: true }),
        applied,
    )

    const legacy = { ...snapshot, reasoningSurvivesPrefix: undefined }
    const oldReplay = replayPlanSnapshot(turns, legacy, reasoningFirst, { allowRegrown: true })
    assert.ok(oldReplay)
    assert.ok(oldReplay.flatMap((entry) => entry.items).every((item) => !selected.has(item.key)))
})

test("OpenCode prefix retains selected native tool results without changing other hosts", () => {
    const turns: Turn[] = []
    for (let index = 0; index < 6; index++) {
        turns.push(
            turn(
                `tool-user-${index}`,
                "user",
                [textItem(`tool-user-${index}`, `Request ${index}`)],
                index * 2 + 1,
            ),
        )
        turns.push(
            turn(
                `tool-assistant-${index}`,
                "assistant",
                [
                    toolItem(
                        `tool-assistant-${index}`,
                        "read",
                        `Specific tool output ${index} ` + "result ".repeat(80),
                        { filePath: `src/file-${index}.ts` },
                    ),
                ],
                index * 2 + 2,
            ),
        )
    }
    const options = inputs({
        contextLimit: 100_000,
        triggerTokens: 1,
        targetTokens: 650,
        force: true,
        recentToolResultBudgetTokens: 650,
        prefixSummaryAllowed: true,
        prefixSummary: "## Decisions\n- Keep current work.\n## Next step\n- Continue.",
    })
    const plan = buildPlan(turns, { ...options, preservePrefixBudgets: true }, spec)
    assert.ok(plan?.requiresCustomCompaction)
    const selected = new Set(plan.preservedToolCallIds)
    assert.ok(selected.size > 0)
    assert.equal(plan.toolSurvivesPrefix, true)
    const output = transformTurns(turns, plan.rawTailStartIndex, plan, spec)
    const retained = output
        .flatMap((entry) => entry.items)
        .filter((item) => item.kind === "tool" && selected.has(item.callId))
    assert.equal(
        retained.length,
        selected.size,
        JSON.stringify({
            selected: [...selected],
            stages: plan.stages.map((stage) => [stage.name, stage.status]),
            emitted: output
                .flatMap((entry) => entry.items)
                .filter((item) => item.kind === "tool")
                .map((item) => item.callId),
        }),
    )
    assert.equal(plan.afterPruneTokens, codec.estimateTurns(output) + plan.overheadTokens)
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), spec, { allowRegrown: true }),
        output,
    )
    const otherHost = buildPlan(turns, options, spec)
    assert.ok(otherHost?.requiresCustomCompaction)
    assert.equal(otherHost.toolSurvivesPrefix, false)
    assert.ok(
        transformTurns(turns, otherHost.rawTailStartIndex, otherHost, spec)
            .flatMap((entry) => entry.items)
            .every((item) => item.kind !== "tool" || !selected.has(item.callId)),
    )
})

test("production reasoning stage does not prune or regrow when a forced plan already meets target", () => {
    const turns = [
        turn("u-old", "user", [textItem("u-old", "Start task")], 1),
        turn(
            "a-old",
            "assistant",
            [reasoningItem("a-old", "Useful old reasoning"), textItem("a-old", "Decision")],
            2,
        ),
        turn("u-middle", "user", [textItem("u-middle", "Check task")], 3),
        turn("a-middle", "assistant", [textItem("a-middle", "Still working")], 4),
        turn("u-new", "user", [textItem("u-new", "Continue task")], 5),
        turn("a-new", "assistant", [reasoningItem("a-new", "Current reasoning")], 6),
    ]
    const productionStage: LadderSpec = { ...spec, stages: [reasoningStage] }
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 100_000,
            targetTokens: 50_000,
            triggerTokens: 1,
            force: true,
            recentReasoningBudgetTokens: 20_000,
        }),
        productionStage,
    )
    assert.ok(plan)
    const transformed = transformTurns(turns, plan.rawTailStartIndex, plan, productionStage)
    assert.ok(
        transformed
            .find((entry) => entry.key === "a-old")
            ?.items.some((item) => item.kind === "reasoning"),
    )
    assert.deepEqual(
        replayPlanSnapshot(turns, toPlanSnapshot(plan), productionStage, { allowRegrown: true }),
        transformed,
    )
})

test("a tail token budget caps the raw tail even when the last user turns span far more", () => {
    // One long tool loop after a user turn: the count-based tail keeps the
    // whole loop raw and leaves nothing to compact. With a budget the ceiling
    // wins, landing on a whole assistant turn inside the loop.
    const turns: Turn[] = [turn("u-1", "user", [textItem("u-1", "start the loop")], 1)]
    for (let index = 0; index < 40; index++) {
        turns.push(
            turn(
                `a-${index}`,
                "assistant",
                [toolItem(`a-${index}`, "bash", `loop output ${index} `.repeat(200))],
                2 + index,
            ),
        )
    }
    turns.push(turn("u-2", "user", [textItem("u-2", "keep going")], 100))
    for (let index = 40; index < 80; index++) {
        turns.push(
            turn(
                `a-${index}`,
                "assistant",
                [toolItem(`a-${index}`, "bash", `loop output ${index} `.repeat(200))],
                2 + index,
            ),
        )
    }
    const budget = { floor: 2_000, ceiling: 6_000 }
    const countBased = buildPlan(turns, inputs({ contextLimit: 30_000, force: true }), spec)
    const budgeted = buildPlan(
        turns,
        inputs({ contextLimit: 30_000, force: true, tailBudgetTokens: budget }),
        spec,
    )
    // Two user turns back is the whole session: nothing left to compact, so
    // the count-based tail produces no plan at all — the original no-op loop.
    assert.equal(countBased, null)
    assert.ok(budgeted)
    assert.ok(budgeted.rawTailStartIndex > 41, "the ceiling must cut inside the second loop")
    assert.equal(budgeted.rawTailItemBoundary, undefined, "turns are never split by the budget")
    const tail = turns.slice(budgeted.rawTailStartIndex)
    assert.ok(codec.estimateTurns(tail) <= budget.ceiling)
    assert.ok(
        codec.estimateTurns(tail) + codec.estimateTurns([turns[budgeted.rawTailStartIndex - 1]]) >
            budget.ceiling,
    )
})

test("the budget opens the tail on a user turn when one lies inside the floor band", () => {
    const turns: Turn[] = [turn("u-1", "user", [textItem("u-1", "older")], 1)]
    for (let index = 0; index < 20; index++) {
        turns.push(
            turn(
                `a-${index}`,
                "assistant",
                [textItem(`a-${index}`, "reply ".repeat(200))],
                2 + index,
            ),
        )
    }
    turns.push(turn("u-2", "user", [textItem("u-2", "recent question")], 50))
    turns.push(turn("a-last", "assistant", [textItem("a-last", "recent answer ".repeat(100))], 51))
    const plan = buildPlan(
        turns,
        inputs({
            contextLimit: 30_000,
            force: true,
            tailBudgetTokens: { floor: 300, ceiling: 3_000 },
        }),
        spec,
    )
    assert.ok(plan)
    assert.equal(turns[plan.rawTailStartIndex]?.key, "u-2")
})

test("selection takes the largest assistant turn first, regardless of age", () => {
    // Size is the only ranking signal: summarizing a turn costs one LLM call
    // either way, so the call must buy the most tokens available. Age used to
    // weight this, which let a small old turn outrank a large recent one.
    // Oldest turn is the smallest here, so age order is the inverse of size.
    const turns: Turn[] = [
        turn("u-1", "user", [textItem("u-1", "start")], 1),
        turn("a-small-old", "assistant", [textItem("a-small-old", "small ".repeat(2_000))], 2),
        turn("u-2", "user", [textItem("u-2", "next")], 3),
        turn("a-large-new", "assistant", [textItem("a-large-new", "large ".repeat(20_000))], 4),
        turn("u-3", "user", [textItem("u-3", "keep going")], 5),
        turn("a-tail", "assistant", [textItem("a-tail", "tail")], 6),
        turn("u-4", "user", [textItem("u-4", "latest")], 7),
    ]

    // A target the largest turn alone can satisfy, so selection stops there.
    const plan = buildPlan(turns, inputs({ contextLimit: 60_000, force: true }), spec)
    assert.ok(plan)
    const selected = plan.summaryJobs.map((job) => job.rangeStartMessageId)
    assert.deepEqual(selected, ["a-large-new"])
})

test("a collapse cap stops one pass early and leaves the rest above target", () => {
    // The cap bounds how much history a single pass may replace with
    // summaries. A pass that cannot reach the target within it stops there,
    // which is what hands the remainder to the host's own compaction.
    const turns: Turn[] = [turn("u-1", "user", [textItem("u-1", "start")], 1)]
    for (let index = 0; index < 10; index++) {
        turns.push(
            turn(
                `a-${index}`,
                "assistant",
                [textItem(`a-${index}`, `detail ${index} `.repeat(4_000))],
                2 + index * 2,
            ),
        )
        turns.push(
            turn(
                `u-${index + 2}`,
                "user",
                [textItem(`u-${index + 2}`, `next ${index}`)],
                3 + index * 2,
            ),
        )
    }
    const options = {
        contextLimit: 30_000,
        force: true,
        targetRatio: 0.05,
        prefixSummaryAllowed: false,
    }

    const uncapped = buildPlan(turns, inputs(options), spec)
    const capped = buildPlan(turns, inputs({ ...options, collapsePercent: 20 }), spec)
    assert.ok(uncapped && capped)

    // A fifth of the collapsible prefix, so a couple of turns at most.
    assert.ok(capped.assistantSummaryKeys.length >= 1)
    assert.ok(capped.assistantSummaryKeys.length <= 2)
    assert.ok(
        capped.assistantSummaryKeys.length < uncapped.assistantSummaryKeys.length,
        "the cap must bind before the uncapped pass stops",
    )
    assert.ok(
        capped.afterPruneTokens > capped.targetTokens,
        "a capped pass leaves the target unmet rather than collapsing further",
    )
})
