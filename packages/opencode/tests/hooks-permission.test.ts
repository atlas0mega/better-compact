import assert from "node:assert/strict"
import test from "node:test"
import { mkdtempSync, writeFileSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"
import type { PluginConfig } from "../lib/config"
import { archiveBoundaryDelta, loadArchiveCatalog } from "../lib/boundary/archive-catalog"
import { openCodeCodec } from "../lib/codec"
import { buildBoundaryContextPlan, processBoundaryTransform } from "../lib/boundary"
import {
    createChatMessageHandler,
    createChatMessageTransformHandler,
    createCommandExecuteHandler,
    createEventHandler,
    createSystemPromptHandler,
    createTextCompleteHandler,
} from "../lib/hooks"
import { Logger } from "../lib/logger"
import { sendIgnoredMessage } from "../lib/ui/notification"
import { getLastUserMessage } from "../lib/messages/query"
import {
    createRuntimeState,
    createSessionState,
    saveSessionState,
    type RuntimeState,
    type WithParts,
} from "../lib/state"

function buildConfig(permission: "allow" | "ask" | "deny" = "allow"): PluginConfig {
    return {
        enabled: true,
        autoUpdate: false,
        debug: false,
        commands: {
            enabled: true,
        },
        compaction: {
            automatic: true,
            preset: "light",
            summaryEffort: "inherit",
            custom: {
                triggerPercent: 85,
                targetPercent: 35,
                recentToolTokens: 40_000,
                summarizerConcurrency: 4,
            },
        },
        experimental: {
            allowSubAgents: false,
        },
        compress: {
            permission,
        },
    }
}

function buildMessage(id: string, role: "user" | "assistant", text: string): WithParts {
    return {
        info: {
            id,
            role,
            sessionID: "session-1",
            agent: "assistant",
            time: { created: 1 },
        } as WithParts["info"],
        parts: [
            {
                id: `${id}-part`,
                messageID: id,
                sessionID: "session-1",
                type: "text",
                text,
            },
        ],
    }
}

function buildUserMessage(
    id: string,
    text: string,
    created: number,
    sessionID = "session-1",
): WithParts {
    return {
        info: {
            id,
            role: "user",
            sessionID,
            agent: "assistant",
            model: { providerID: "anthropic", modelID: "claude-test" },
            time: { created },
        } as WithParts["info"],
        parts: [
            {
                id: `${id}-part`,
                messageID: id,
                sessionID,
                type: "text",
                text,
            },
        ],
    }
}

function buildAssistantToolMessage(
    id: string,
    created: number,
    reportedTokens?: number,
    sessionID = "session-1",
): WithParts {
    return {
        info: {
            id,
            role: "assistant",
            sessionID,
            agent: "assistant",
            time: { created },
            ...(reportedTokens
                ? {
                      tokens: {
                          total: reportedTokens,
                          input: reportedTokens - 1,
                          output: 1,
                          reasoning: 0,
                          cache: { read: 0, write: 0 },
                      },
                  }
                : {}),
        } as WithParts["info"],
        parts: [
            {
                id: `${id}-tool`,
                messageID: id,
                sessionID,
                type: "tool",
                callID: `${id}-call`,
                tool: "read",
                state: {
                    status: "completed",
                    input: { filePath: "src/app.ts" },
                    output: "tool output ".repeat(500),
                    title: "read",
                    metadata: {},
                    time: { start: 1, end: 2 },
                },
            } as any,
        ],
    }
}

function buildReducibleOldToolMessage(id: string, created: number, sessionID: string): WithParts {
    const message = buildAssistantToolMessage(id, created, undefined, sessionID)
    const tool = message.parts[0]
    if (tool.type !== "tool" || tool.state.status !== "completed") throw new Error("fixture")
    tool.state.output = "huge tool output ".repeat(12_000)
    return message
}

async function waitFor(condition: () => boolean, timeoutMs = 10_000): Promise<void> {
    const started = Date.now()
    while (!condition()) {
        if (Date.now() - started > timeoutMs) {
            throw new Error("Timed out waiting for condition")
        }
        await new Promise((resolve) => setTimeout(resolve, 25))
    }
}

test("system prompt handler caches full model context for percentage thresholds", async () => {
    const logger = new Logger(false)
    const runtime = createRuntimeState({}, logger)
    const handler = createSystemPromptHandler(runtime, logger, buildConfig("deny"))

    await handler(
        {
            sessionID: "session-1",
            model: {
                limit: {
                    context: 200000,
                    output: 131072,
                },
            },
        } as any,
        { system: ["base system"] },
    )

    assert.equal(runtime.get("session-1").modelContextLimit, 200000)
})

test("chat message transform strips hallucinated tags even when compress is denied", async () => {
    const logger = new Logger(false)
    const config = buildConfig("deny")
    const client = { session: { get: async () => ({}) }, provider: { list: async () => [] } }
    const runtime = createRuntimeState(client, logger)
    const handler = createChatMessageTransformHandler(client as any, runtime, logger, config, {
        global: undefined,
        agents: {},
    })
    const output = {
        messages: [buildMessage("assistant-1", "assistant", "alpha <dcp>beta</dcp> omega")],
    }

    await handler({}, output)

    assert.equal(output.messages[0]?.parts[0]?.type, "text")
    assert.equal((output.messages[0]?.parts[0] as any).text, "alpha  omega")
})

test("chat message transform drops messages without info instead of crashing", async () => {
    const logger = new Logger(false)
    const config = buildConfig("deny")
    const client = { session: { get: async () => ({}) }, provider: { list: async () => [] } }
    const runtime = createRuntimeState(client, logger)
    const handler = createChatMessageTransformHandler(client as any, runtime, logger, config, {
        global: undefined,
        agents: {},
    })
    const output = {
        messages: [
            {
                role: "user",
                time: 1,
                parts: [
                    {
                        type: "text",
                        text: "Carica le skill di laravel",
                    },
                ],
            } as any,
        ],
    }

    await handler({}, output as any)

    assert.equal(runtime.peek("session-1"), undefined)
    assert.equal(output.messages.length, 0)
})

function buildOverTriggerConversation(sessionId: string): WithParts[] {
    const withSession = (message: WithParts): WithParts => {
        message.info.sessionID = sessionId
        for (const part of message.parts) part.sessionID = sessionId
        return message
    }
    const big = buildReducibleOldToolMessage("assistant-big", 2, sessionId)
    return [
        withSession(buildUserMessage("user-1", "old user request", 1)),
        withSession(big),
        withSession(buildUserMessage("user-2", "middle user request", 3)),
        withSession(buildMessage("assistant-2", "assistant", "middle assistant response")),
        withSession(buildUserMessage("user-3", "latest user request", 5)),
    ]
}

function buildProfileEscalationConversation(sessionId: string): WithParts[] {
    const messages = [
        buildUserMessage("profile-old-user", "Important instruction ".repeat(2_000), 1, sessionId),
    ]
    for (let index = 0; index < 8; index++) {
        const assistant = buildMessage(
            `profile-assistant-${index}`,
            "assistant",
            `progress ${index} ${"x".repeat(4_000)}`,
        )
        assistant.info.sessionID = sessionId
        for (const part of assistant.parts) part.sessionID = sessionId
        messages.push(assistant)
    }
    messages.push(buildUserMessage("profile-middle-user", "middle request", 20, sessionId))
    const tail = buildMessage("profile-tail", "assistant", "recent response")
    tail.info.sessionID = sessionId
    for (const part of tail.parts) part.sessionID = sessionId
    messages.push(tail, buildUserMessage("profile-latest-user", "current request", 22, sessionId))
    return messages
}

function profileEscalationConfig(prefixSummary: boolean, collapsePercent: number): PluginConfig {
    const config = buildConfig("allow")
    config.compaction = {
        automatic: true,
        preset: "custom",
        summaryEffort: "off",
        custom: {
            triggerPercent: 1,
            targetPercent: 1,
            recentToolTokens: 0,
            summarizerConcurrency: 1,
            prefixSummary,
            collapsePercent,
        },
    }
    return config
}

function transformClient(contextLimit: number, toasts: unknown[] = []) {
    return {
        session: { get: async () => ({ data: { parentID: null } }) },
        provider: {
            list: async () => [
                {
                    id: "anthropic",
                    models: { "claude-test": { limit: { context: contextLimit } } },
                },
            ],
        },
        tui: {
            showToast: async (input: unknown) => {
                toasts.push(input)
            },
        },
    }
}

function transformHandler(
    client: any,
    runtime: RuntimeState,
    config: PluginConfig,
    directory: string,
) {
    return createChatMessageTransformHandler(
        client,
        runtime,
        new Logger(false),
        config,
        { global: undefined, agents: {} },
        directory,
    )
}

function withProviderUsage(messages: WithParts[], total: number): WithParts[] {
    const last = [...messages].reverse().find((message) => message.info.role === "assistant")
    assert.ok(last)
    Object.assign(last.info, {
        tokens: { total, input: total - 1, output: 1, reasoning: 0, cache: { read: 0, write: 0 } },
    })
    return messages
}

async function finishAutoTurn(
    client: any,
    runtime: RuntimeState,
    config: PluginConfig,
    directory: string,
    messages: WithParts[],
    sessionID: string,
): Promise<void> {
    client.session.messages = async () => ({ data: messages })
    await createEventHandler(runtime, new Logger(false), client, config, directory, {
        global: undefined,
        agents: {},
    })({
        event: { type: "session.idle", properties: { sessionID } },
    })
}

test("idle auto-compaction uses the TUI provider count, not a huge archived transcript", async () => {
    const sessionID = `ses-idle-usage-${Date.now()}`
    let messages = [
        ...buildOverTriggerConversation(sessionID),
        buildAssistantToolMessage("idle-usage-33", 6, 3_300, sessionID),
    ]
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-idle-usage-"))
    const client = {
        ...transformClient(10_000),
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
        },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const state = runtime.get(sessionID)
    const event = createEventHandler(runtime, logger, client, config, directory, {
        global: undefined,
        agents: {},
    })
    const idle = { event: { type: "session.idle", properties: { sessionID } } }
    await event(idle)
    assert.equal(state.boundary.activePlan, null)
    assert.equal(state.boundary.automaticCheck?.reason, "below_trigger")
    assert.equal(state.boundary.automaticCheck?.seam, "idle")
    await event(idle)
    assert.equal(state.boundary.activePlan, null)

    messages = [...messages, buildAssistantToolMessage("idle-usage-90", 7, 9_000, sessionID)]
    await event(idle)
    assert.ok(state.boundary.activePlan)
    assert.equal(state.boundary.automaticCheck?.seam, "idle")
    assert.equal(state.boundary.automaticCheck?.reason, "planned")
    const output = { messages: structuredClone(messages) }
    await transformHandler(client, runtime, config, directory)({}, output)
    assert.ok(output.messages.some((message) => message.info.id.startsWith("msg_better_compact_")))
    assert.equal(state.boundary.automaticCheck?.seam, "pre_request")
})

test("missing model limits report a reason at idle and before the next provider request", async () => {
    const sessionID = `ses-missing-limit-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionID), 9_000)
    const directory = mkdtempSync(join(tmpdir(), "better-compact-missing-limit-"))
    const client = {
        ...transformClient(10_000),
        provider: { list: async () => [] },
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const config = buildConfig("allow")
    const output = { messages: structuredClone(messages) }
    await transformHandler(client, runtime, config, directory)({}, output)
    const state = runtime.get(sessionID)
    assert.equal(state.boundary.automaticCheck?.reason, "model_limit_unknown")
    assert.equal(state.boundary.activePlan, null)
    assert.deepEqual(output.messages, messages)
    await finishAutoTurn(client, runtime, config, directory, messages, sessionID)
    assert.equal(state.boundary.automaticCheck?.reason, "model_limit_unknown")
    assert.equal(state.boundary.activePlan, null)
})

test("the next provider request compacts an overflowing tool loop even without an idle event", async () => {
    const sessionID = `ses-overflow-guard-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionID), 3_300)
    const big = messages[1].parts[0] as any
    big.state.output = "large completed tool result ".repeat(8_000)
    const client = transformClient(10_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const output = { messages }
    await transformHandler(
        client,
        runtime,
        buildConfig("allow"),
        mkdtempSync(join(tmpdir(), "better-compact-overflow-")),
    )({}, output)
    assert.ok(runtime.get(sessionID).boundary.activePlan)
    assert.ok(output.messages.some((item) => item.info.id.startsWith("msg_better_compact_")))
    assert.ok(openCodeCodec.estimateTurns(openCodeCodec.encode(output.messages)) < 10_000)
})

test("a stale saved plan is rebuilt before an overflowing request rather than silently replayed", async () => {
    const sessionID = `ses-stale-overflow-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionID), 9_000)
    const client = transformClient(10_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-stale-overflow-"))
    await finishAutoTurn(client, runtime, config, directory, messages, sessionID)
    assert.ok(runtime.get(sessionID).boundary.activePlan)
    const changed = structuredClone(messages)
    const oldTool = changed[1].parts[0] as any
    oldTool.state.output = "Revised historical tool output ".repeat(8_000)
    const output = { messages: changed }
    await transformHandler(client, runtime, config, directory)({}, output)
    assert.ok(openCodeCodec.estimateTurns(openCodeCodec.encode(output.messages)) < 10_000)
    assert.ok(output.messages.some((item) => item.info.id.startsWith("msg_better_compact_")))
    assert.notEqual(runtime.get(sessionID).boundary.automaticCheck?.reason, "plan_replayed")
})

test("an irreducible oversized user turn stops the provider request", async () => {
    const sessionID = `ses-irreducible-${Date.now()}`
    const messages = [
        buildUserMessage("giant-user", "Current instruction ".repeat(8_000), 1, sessionID),
    ]
    const client = transformClient(10_000)
    const runtime = createRuntimeState(client, new Logger(false))
    await assert.rejects(
        transformHandler(
            client,
            runtime,
            buildConfig("allow"),
            mkdtempSync(join(tmpdir(), "better-compact-irreducible-")),
        )({}, { messages }),
        /provider request was stopped/,
    )
    assert.equal(runtime.get(sessionID).boundary.automaticCheck?.reason, "overflow_unresolved")
})

test("idle auto-compaction stores a plan that the next transform replays", async () => {
    const sessionId = `ses-transform-allow-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionId), 9_000)
    const toasts: unknown[] = []
    const client = transformClient(10_000, toasts)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionId)
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-transform-"))
    const handler = transformHandler(client, runtime, config, directory)
    const output = { messages }

    await finishAutoTurn(client, runtime, config, directory, messages, sessionId)
    await handler({}, output)

    assert.ok(state.boundary.activePlan)
    assert.ok(messages.some((item) => item.info.id.startsWith("msg_better_compact_")))
    assert.ok(!messages.some((item) => item.parts.some((part) => part.type === "tool")))
    assert.equal(toasts.length, 1)
    assert.equal(state.boundary.lastPlannedUsageMessageId, "assistant-2")
})

test("an above-trigger provider reading is consumed once but a new response is checked before its next request", async () => {
    const sessionID = `ses-consumed-usage-${Date.now()}`
    const toasts: unknown[] = []
    const client = transformClient(100_000, toasts)
    const runtime = createRuntimeState(client, new Logger(false))
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-consumed-usage-"))
    const handler = transformHandler(client, runtime, config, directory)
    const messages = [
        buildUserMessage("user-1", "old request", 1, sessionID),
        buildReducibleOldToolMessage("assistant-1", 2, sessionID),
        buildUserMessage("user-2", "middle request", 3, sessionID),
        buildAssistantToolMessage("assistant-2", 4, 90_000, sessionID),
        buildUserMessage("user-3", "latest request", 5, sessionID),
    ]
    await finishAutoTurn(client, runtime, config, directory, messages, sessionID)
    const state = runtime.get(sessionID)
    assert.equal(state.boundary.lastPlannedUsageMessageId, "assistant-2")
    await handler({}, { messages: structuredClone(messages) })
    assert.equal(toasts.length, 1, "the same provider response must not start a second job")

    const advanced = [...messages, buildAssistantToolMessage("assistant-3", 6, 90_000, sessionID)]
    const previousChecks = state.boundary.automaticCheck?.count ?? 0
    await handler({}, { messages: structuredClone(advanced) })
    assert.equal(state.boundary.lastPlannedUsageMessageId, "assistant-3")
    assert.ok((state.boundary.automaticCheck?.count ?? 0) > previousChecks)
    assert.equal(state.boundary.automaticCheck?.reason, "plan_replayed")
    assert.equal(
        toasts.length,
        1,
        "a checked response with no new eligible boundary replays without a new job",
    )
    await handler({}, { messages: structuredClone(advanced) })
    assert.equal(toasts.length, 1, "the second response must also be consumed exactly once")
})

test("above-trigger assistant/tool continuation archives a long loop before the next provider call", async () => {
    const sessionID = `ses-agentic-tail-${Date.now()}`
    const directory = mkdtempSync(join(tmpdir(), "better-compact-agentic-tail-"))
    const client = transformClient(100_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const config = buildConfig("allow")
    config.compaction.preset = "custom"
    config.compaction.custom = {
        triggerPercent: 66,
        targetPercent: 18,
        recentToolTokens: 0,
        summarizerConcurrency: 4,
        collapsePercent: 45,
        prefixSummary: true,
    }
    const initial = [
        buildUserMessage("u-1", "Initial goal", 1, sessionID),
        buildAssistantToolMessage("a-1", 2, undefined, sessionID),
        buildUserMessage("u-2", "Earlier correction", 3, sessionID),
        buildAssistantToolMessage("a-2", 4, undefined, sessionID),
        buildUserMessage("u-active", "Continue the active implementation", 5, sessionID),
        buildAssistantToolMessage("a-active", 6, 90_000, sessionID),
    ]
    await finishAutoTurn(client, runtime, config, directory, initial, sessionID)
    const originalBoundary = runtime.get(sessionID).boundary.activePlan?.rawTailStartMessageId
    assert.ok(originalBoundary, "first turn must have committed a plan")
    const grown = structuredClone(initial)
    for (let index = 0; index < 50; index++) {
        const item = buildAssistantToolMessage(
            `a-loop-${index}`,
            7 + index,
            index === 49 ? 90_000 : undefined,
            sessionID,
        )
        const part = item.parts[0]
        if (part.type === "tool" && part.state.status === "completed")
            part.state.output =
                `Unique historical tool ${index} ` + "bulky loop output ".repeat(750)
        grown.push(item)
    }
    const output = { messages: structuredClone(grown) }
    await transformHandler(client, runtime, config, directory)({}, output)
    const plan = runtime.get(sessionID).boundary.activePlan
    assert.ok(plan)
    assert.notEqual(plan.rawTailStartMessageId, originalBoundary)
    assert.match(plan.rawTailStartMessageId, /^a-loop-/)
    const archived = await loadArchiveCatalog(directory, sessionID)
    assert.ok(
        archived.entries.length >= 2,
        JSON.stringify({
            originalBoundary,
            newBoundary: plan.rawTailStartMessageId,
            planArchiveGeneration: plan.archiveGeneration,
            outcome: runtime.get(sessionID).boundary.lastAutomaticCheck?.reason,
            entries: archived.entries.map((entry) => ({
                id: entry.id,
                count: Object.keys(entry.fingerprints).length,
            })),
        }),
    )
    assert.doesNotMatch(JSON.stringify(output.messages), /Unique historical tool 0 /)
    assert.match(JSON.stringify(output.messages), /Continue the active implementation/)
    assert.match(
        JSON.stringify(grown),
        /Unique historical tool 0 /,
        "native history must stay unchanged",
    )
})

test("auto transform path honors the configured compaction profile", async () => {
    const sessionId = `ses-transform-profile-${Date.now()}`
    // ~17K estimated tokens on a 200K limit: far below the default 85%
    // trigger, above a custom 5% one.
    const messages = withProviderUsage(buildOverTriggerConversation(sessionId), 16_000)
    const client = transformClient(200_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionId)
    const config = buildConfig("allow")
    config.compaction = {
        automatic: true,
        preset: "custom",
        summaryEffort: "inherit",
        custom: {
            triggerPercent: 5,
            targetPercent: 3,
            recentToolTokens: 0,
            summarizerConcurrency: 4,
        },
    }
    await finishAutoTurn(
        client,
        runtime,
        config,
        mkdtempSync(join(tmpdir(), "better-compact-transform-")),
        messages,
        sessionId,
    )

    assert.ok(
        state.boundary.activePlan,
        "custom low trigger must produce a plan where the default would not",
    )
    assert.equal(state.boundary.activePlan.triggerTokens, Math.floor(200_000 * 0.05))

    const controlSessionId = `${sessionId}-control`
    const untouched = withProviderUsage(buildOverTriggerConversation(controlSessionId), 16_000)
    const controlClient = transformClient(200_000)
    const controlRuntime = createRuntimeState(controlClient, new Logger(false))
    await finishAutoTurn(
        controlClient,
        controlRuntime,
        buildConfig("allow"),
        mkdtempSync(join(tmpdir(), "better-compact-transform-")),
        untouched,
        controlSessionId,
    )

    assert.equal(
        controlRuntime.get(controlSessionId).boundary.activePlan,
        null,
        "default 85% trigger must not fire at ~8% usage",
    )
})

test("automatic compaction honors the prefix-summary opt-in and assistant collapse cap", async () => {
    const plans = [] as NonNullable<ReturnType<RuntimeState["get"]>["boundary"]["activePlan"]>[]
    for (const [prefixSummary, collapsePercent] of [
        [false, 10],
        [false, 75],
        [true, 10],
    ] as const) {
        const sessionId = `ses-auto-profile-${prefixSummary}-${collapsePercent}-${Date.now()}`
        const messages = withProviderUsage(buildProfileEscalationConversation(sessionId), 40_000)
        const client = transformClient(50_000)
        const runtime = createRuntimeState(client, new Logger(false))
        await finishAutoTurn(
            client,
            runtime,
            profileEscalationConfig(prefixSummary, collapsePercent),
            mkdtempSync(join(tmpdir(), "better-compact-profile-auto-")),
            messages,
            sessionId,
        )
        const plan = runtime.get(sessionId).boundary.activePlan
        assert.ok(plan)
        plans.push(plan)
    }

    assert.ok(!plans[0].stages.some((stage) => stage.name === "prefix-summary"))
    assert.equal(plans[0].assistantSummaryKeys?.length, 1)
    assert.ok((plans[1].assistantSummaryKeys?.length ?? 0) > 1)
    assert.ok(plans[2].stages.some((stage) => stage.name === "prefix-summary"))
    assert.equal(plans[2].assistantSummaryKeys?.length, 0, "prefix absorbs the per-turn cache")
})

test("manual compaction honors the prefix-summary opt-in and assistant collapse cap", async () => {
    const plans = [] as NonNullable<ReturnType<RuntimeState["get"]>["boundary"]["activePlan"]>[]
    for (const [prefixSummary, collapsePercent] of [
        [false, 10],
        [false, 75],
        [true, 10],
    ] as const) {
        const sessionId = `ses-manual-profile-${prefixSummary}-${collapsePercent}-${Date.now()}`
        const messages = buildProfileEscalationConversation(sessionId)
        const client = {
            session: {
                get: async () => ({ data: { parentID: null } }),
                messages: async () => ({ data: messages }),
                prompt: async () => ({ data: true }),
            },
        }
        const runtime = createRuntimeState(client, new Logger(false))
        const state = runtime.get(sessionId)
        state.modelContextLimit = 50_000
        await createCommandExecuteHandler(
            client as any,
            runtime,
            new Logger(false),
            profileEscalationConfig(prefixSummary, collapsePercent),
            mkdtempSync(join(tmpdir(), "better-compact-profile-manual-")),
            { global: undefined, agents: {} },
        )({ command: "better-compact", sessionID: sessionId, arguments: "" }, { parts: [] })
        await waitFor(() => state.boundary.job?.status === "completed")
        assert.ok(state.boundary.activePlan)
        plans.push(state.boundary.activePlan)
    }

    assert.ok(!plans[0].stages.some((stage) => stage.name === "prefix-summary"))
    assert.equal(plans[0].assistantSummaryKeys?.length, 1)
    assert.ok((plans[1].assistantSummaryKeys?.length ?? 0) > 1)
    assert.ok(plans[2].stages.some((stage) => stage.name === "prefix-summary"))
    assert.equal(plans[2].assistantSummaryKeys?.length, 0, "prefix absorbs the per-turn cache")
})

test("manual TUI archive handoff uses its default variant independent of turn-summary effort", async () => {
    const sessionId = `ses-summary-model-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionId)
    const summary = [
        "## Decisions",
        "- Completed the requested work.",
        "## Files & Symbols",
        "- src/app.ts",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- (none)",
        "## Constraints",
        "- Preserve the contract.",
        "## Next step",
        "- Continue implementation.",
    ].join("\n")
    const scratchPrompts: any[] = []
    const client = {
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
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            create: async ({ body }: any) => {
                assert.deepEqual(body.model, { providerID: "openai", id: "gpt-6-luna" })
                return { data: { id: `scratch-${sessionId}` } }
            },
            prompt: async (input: any) => {
                if (input.body.noReply) return { data: true }
                scratchPrompts.push(input)
                return { data: { parts: [{ type: "text", text: summary }] } }
            },
            delete: async () => ({ data: true }),
        },
    }
    const config = profileEscalationConfig(true, 10)
    config.compaction.summaryEffort = "high"
    config.compaction.summaryModel = "openai/gpt-6-luna"
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionId)
    state.modelContextLimit = 50_000
    const handler = createChatMessageHandler(
        client as any,
        runtime,
        new Logger(false),
        config,
        mkdtempSync(join(tmpdir(), "better-compact-summary-model-")),
        { global: undefined, agents: {} },
    )
    await handler(
        {
            sessionID: sessionId,
            model: { providerID: "anthropic", modelID: "claude-test" },
            variant: "low",
        },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        summaryVariant: "low",
                        summaryProviderID: "anthropic",
                        summaryModelID: "claude-test",
                        contextLimit: 50_000,
                    },
                },
            ],
        },
    )
    await waitFor(() => state.boundary.job?.status === "completed")
    assert.ok(scratchPrompts.length > 0)
    for (const prompt of scratchPrompts) {
        assert.deepEqual(prompt.body.model, { providerID: "openai", modelID: "gpt-6-luna" })
        assert.equal(
            prompt.body.variant,
            (prompt.body.parts[0].text as string).includes("Describe only archive")
                ? undefined
                : "high",
        )
    }
    assert.ok((state.boundary.job?.counters.summaryJobsDone ?? 0) > 0)
})

test("manual compaction avoids paying for per-turn summaries it cannot apply", async () => {
    const sessionId = `ses-summary-growth-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionId)
    const config = profileEscalationConfig(false, 10)
    config.compaction.summaryEffort = "high"
    config.compaction.summaryModel = "openai/gpt-6-luna"
    const baseline = buildBoundaryContextPlan(messages, {
        contextLimit: 50_000,
        force: true,
        triggerRatio: 0.01,
        targetRatio: 0.01,
        recentToolResultBudgetTokens: 0,
        prefixSummaryAllowed: false,
        collapsePercent: 10,
        summariesAllowed: true,
    })
    assert.ok(baseline?.summaryJobs.length)
    const verboseSummary = [
        "## Decisions",
        `- ${"Detailed but redundant work. ".repeat(115)}`,
        "## Files & Symbols",
        "- src/app.ts",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- (none)",
        "## Constraints",
        "- Preserve the contract.",
        "## Next step",
        "- Continue implementation.",
    ].join("\n")
    let calls = 0
    const client = {
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "openai",
                            models: {
                                "gpt-6-luna": { variants: { high: {} } },
                            },
                        },
                    ],
                },
            }),
        },
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            create: async () => ({ data: { id: `scratch-${sessionId}` } }),
            prompt: async ({ body }: any) => {
                if (body.noReply) return { data: true }
                calls++
                return { data: { parts: [{ type: "text", text: verboseSummary }] } }
            },
            delete: async () => ({ data: true }),
        },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const state = runtime.get(sessionId)
    state.modelContextLimit = 50_000
    const handler = createChatMessageHandler(
        client as any,
        runtime,
        logger,
        config,
        mkdtempSync(join(tmpdir(), "better-compact-summary-growth-")),
        { global: undefined, agents: {} },
    )
    await handler(
        { sessionID: sessionId, model: { providerID: "anthropic", modelID: "claude-test" } },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        contextLimit: 50_000,
                        summaryProviderID: "anthropic",
                        summaryModelID: "claude-test",
                    },
                },
            ],
        },
    )
    await waitFor(() => state.boundary.job?.status === "completed")
    assert.equal(calls, 0)
    assert.ok(state.boundary.activePlan)
    assert.deepEqual(state.boundary.activePlan?.assistantSummaries, {})
    assert.ok(baseline.afterPruneTokens > 0)
})

test("manual prefix compaction keeps the fallback when the whole handoff is too large", async () => {
    const sessionId = `ses-prefix-chunks-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionId)
    const config = profileEscalationConfig(true, 45)
    config.compaction.summaryEffort = "high"
    config.compaction.summaryModel = "openai/gpt-6-luna"
    let prompts = 0
    let livePrompts = 0
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
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            create: async () => ({ data: { id: `scratch-${sessionId}` } }),
            prompt: async ({ body }: any) => {
                if (body.noReply) return { data: true }
                prompts++
                assert.deepEqual(body.model, { providerID: "openai", modelID: "gpt-6-luna" })
                const request = body.parts[0].text as string
                if (request.includes("Describe only archive")) {
                    assert.equal(body.variant, undefined)
                    return {
                        data: {
                            parts: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        description:
                                            "Implementation archive with prior decisions and test evidence.",
                                    }),
                                },
                            ],
                        },
                    }
                }
                livePrompts++
                assert.equal(body.variant, "high")
                assert.match(
                    request,
                    /Archive evidence:|Chronological chunk|Extracted chronological evidence:/,
                )
                assert.doesNotMatch(request, /Source transcript:\n# Better Compact Raw Transcript/)
                const descriptionIds =
                    request
                        .match(/Required description IDs: ([^\n]+)\./)?.[1]
                        .split(/,\s*/)
                        .filter(Boolean) ?? []
                const handoff = [
                    "## Decisions",
                    "- Completed src/app.ts while keeping the active task intact.",
                    "## Files & Symbols",
                    "- src/app.ts",
                    "## Errors (verbatim)",
                    "- (none)",
                    "## What failed and why",
                    "- (none)",
                    "## Constraints",
                    "- Keep user requirements.",
                    "## Next step",
                    `- Validate the latest implementation. ${"Verbose historical progress. ".repeat(15_000)}`,
                ].join("\n")
                return {
                    data: {
                        parts: [
                            {
                                type: "text",
                                text: descriptionIds.length
                                    ? JSON.stringify({
                                          handoff,
                                          descriptions: Object.fromEntries(
                                              descriptionIds.map((id) => [
                                                  id,
                                                  "Task-area implementation, earlier decisions and verification context.",
                                              ]),
                                          ),
                                      })
                                    : handoff,
                            },
                        ],
                    },
                }
            },
            delete: async () => ({ data: true }),
        },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(sdk, logger)
    const state = runtime.get(sessionId)
    state.modelContextLimit = 50_000
    const directory = mkdtempSync(join(tmpdir(), "better-compact-prefix-chunks-"))
    const handler = createChatMessageHandler(sdk as any, runtime, logger, config, directory, {
        global: undefined,
        agents: {},
    })
    await handler(
        { sessionID: sessionId, model: { providerID: "anthropic", modelID: "claude-test" } },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        contextLimit: 50_000,
                        summaryProviderID: "anthropic",
                        summaryModelID: "claude-test",
                    },
                },
            ],
        },
    )
    await waitFor(() => state.boundary.job?.status === "completed")
    assert.ok(prompts >= 1 && prompts <= 7)
    assert.equal(state.boundary.job?.counters.summaryJobsDone, livePrompts)
    assert.match(state.boundary.activePlan?.prefixSummary ?? "", /Important instruction/)
    assert.doesNotMatch(
        state.boundary.activePlan?.prefixSummary ?? "",
        /Validate the latest implementation/,
    )
    const catalog = await loadArchiveCatalog(directory, sessionId)
    assert.ok(["pending", "ready"].includes(catalog.entries.at(-1)?.status ?? ""))
    assert.ok(catalog.entries.at(-1)?.oversizedSummaryPath)
    assert.ok(state.boundary.activePlan!.afterPruneTokens < state.boundary.activePlan!.beforeTokens)
    assert.ok(
        (state.boundary.activePlan?.preservedPrefixTurnKeys?.length ?? 0) <
            Object.keys(catalog.entries.at(-1)!.fingerprints).length,
        "an archived but rejected handoff must not keep all associated turns native",
    )
    const firstJob = state.boundary.job
    const firstPrompts = livePrompts
    assert.equal(state.boundary.activePlan?.prefixChunkVersion, 2)
    await waitFor(() => !runtime.activeCompaction(sessionId))
    await handler(
        { sessionID: sessionId, model: { providerID: "anthropic", modelID: "claude-test" } },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        contextLimit: 50_000,
                        summaryProviderID: "anthropic",
                        summaryModelID: "claude-test",
                    },
                },
            ],
        },
    )
    await waitFor(
        () => state.boundary.job !== firstJob && state.boundary.job?.status === "completed",
    )
    assert.equal(livePrompts, firstPrompts, "unchanged range must not charge another live handoff")
    assert.equal((await loadArchiveCatalog(directory, sessionId)).entries.length, 1)
    let unexpectedCalls = 0
    const priorRange = state.boundary.activePlan?.rangeHash
    let attemptedRange: string | undefined
    await processBoundaryTransform({
        state,
        logger,
        config,
        directory,
        messages: structuredClone(messages),
        summariesAllowed: true,
        summarizeArchive: async (plan) => {
            unexpectedCalls++
            attemptedRange = plan.rangeHash
            return { ok: false, reason: "invalid_output", calls: 1 }
        },
    })
    assert.equal(
        unexpectedCalls,
        0,
        `automatic replay must not retry the same boundary: prior=${priorRange}, attempted=${attemptedRange}`,
    )
    assert.equal((await loadArchiveCatalog(directory, sessionId)).entries.length, 1)
})

test("auto transform path never prunes when compress permission is deny", async () => {
    const sessionId = `ses-transform-deny-${Date.now()}`
    const messages = buildOverTriggerConversation(sessionId)
    const client = transformClient(10_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionId)
    const handler = transformHandler(
        client,
        runtime,
        buildConfig("deny"),
        mkdtempSync(join(tmpdir(), "better-compact-transform-deny-")),
    )
    const before = JSON.stringify(messages)
    const output = { messages }

    await handler({}, output)

    assert.equal(state.boundary.activePlan, null)
    assert.equal(JSON.stringify(messages), before)
})

test("automatic compaction uses freshly loaded global settings", async () => {
    const client = transformClient(1_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get("session-1")
    const startupConfig = buildConfig("allow")
    const currentConfig = buildConfig("allow")
    currentConfig.compaction.automatic = false
    const handler = createChatMessageTransformHandler(
        client as any,
        runtime,
        new Logger(false),
        startupConfig,
        { global: undefined, agents: {} },
        mkdtempSync(join(tmpdir(), "better-compact-automatic-")),
        () => currentConfig,
    )
    const output = {
        messages: [
            buildUserMessage("user-1", "old request", 1),
            buildAssistantToolMessage("assistant-1", 2),
            buildUserMessage("user-2", "middle request", 3),
            buildAssistantToolMessage("assistant-2", 4),
            buildUserMessage("user-3", "latest request", 5),
        ],
    }

    await handler({}, output)

    assert.equal(state.boundary.activePlan, null)
})

test("changing the target below trigger retires a stale overcompact plan without lowering the trigger", async () => {
    const sessionID = `ses-policy-refresh-${process.pid}-${Date.now()}`
    const config = buildConfig("allow")
    const client = transformClient(100_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionID)
    const directory = mkdtempSync(join(tmpdir(), "better-compact-policy-refresh-"))
    const messages = withProviderUsage(buildOverTriggerConversation(sessionID), 90_000)
    await finishAutoTurn(client, runtime, config, directory, messages, sessionID)
    assert.ok(state.boundary.activePlan)
    const oldTarget = state.boundary.activePlan.targetTokens
    const archiveCount = (await loadArchiveCatalog(directory, sessionID)).entries.length
    assert.equal(archiveCount, 1)

    // The resumed provider usage is below the unchanged 85% trigger; settings
    // changes still must not continue replaying the older, tighter snapshot.
    withProviderUsage(messages, 3_000)
    config.compaction = {
        ...config.compaction,
        preset: "custom",
        custom: {
            triggerPercent: 85,
            targetPercent: 25,
            recentToolTokens: 40_000,
            recentReasoningTokens: 28_000,
            summarizerConcurrency: 4,
        },
    }
    const output = { messages: structuredClone(messages) }
    await transformHandler(client, runtime, config, directory)({}, output)

    assert.equal(oldTarget, 35_000)
    assert.equal(state.boundary.activePlan, null)
    assert.deepEqual(output.messages, messages)
    assert.equal(state.boundary.automaticCheck?.reason, "below_trigger")
    assert.equal((await loadArchiveCatalog(directory, sessionID)).entries.length, archiveCount)
})

test("automatic compaction triggers from provider usage when the local estimate is lower", async () => {
    const sessionID = `session-provider-${process.pid}-${Date.now()}`
    const toasts: unknown[] = []
    const client = transformClient(100_000, toasts)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionID)
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-provider-trigger-"))
    const output = {
        messages: [
            buildUserMessage("user-1", "old request", 1, sessionID),
            buildReducibleOldToolMessage("assistant-1", 2, sessionID),
            buildUserMessage("user-2", "middle request", 3, sessionID),
            buildAssistantToolMessage("assistant-2", 4, 90_000, sessionID),
            buildUserMessage("user-3", "latest request", 5, sessionID),
        ],
    }

    await finishAutoTurn(client, runtime, config, directory, output.messages, sessionID)

    assert.ok(state.boundary.activePlan)
    assert.equal(state.boundary.activePlan.beforeTokens, 90_000)
    assert.equal(toasts.length, 1)
})

test("assistant/tool turn compacts at idle before the next provider request without a new user message", async () => {
    const sessionID = `ses-loop-${process.pid}-${Date.now()}`
    const client = transformClient(100_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionID)
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-loop-"))
    const handler = transformHandler(client, runtime, config, directory)
    const messages = [
        buildUserMessage("user-1", "start", 1, sessionID),
        buildReducibleOldToolMessage("assistant-1", 2, sessionID),
        buildUserMessage("user-2", "continue", 3, sessionID),
        buildAssistantToolMessage("assistant-2", 4, 10_000, sessionID),
        buildUserMessage("user-3", "begin tool loop", 5, sessionID),
    ]
    await handler({}, { messages: structuredClone(messages) })
    assert.equal(state.boundary.activePlan, null)
    const output = {
        messages: [...messages, buildAssistantToolMessage("assistant-3", 6, 90_000, sessionID)],
    }
    await finishAutoTurn(client, runtime, config, directory, output.messages, sessionID)
    assert.ok(state.boundary.activePlan)
    await handler({}, output)
    assert.equal(state.boundary.automaticCheck?.reason, "plan_replayed")
    assert.ok(output.messages.some((message) => message.info.id.startsWith("msg_better_compact_")))
    assert.equal(output.messages.at(-1)?.info.id, "assistant-3")
})

test("a validated live handoff keeps first-archive wording before the provider request", async () => {
    const sessionID = `ses-pre-provider-handoff-${Date.now()}`
    const directory = mkdtempSync(join(tmpdir(), "better-compact-pre-provider-handoff-"))
    const messages = [
        buildUserMessage("user-old", "Secret legacy widget policy must apply.", 1, sessionID),
        buildAssistantToolMessage("assistant-old", 2, undefined, sessionID),
        buildUserMessage("user-middle", "Check widget tests carefully.", 3, sessionID),
        buildAssistantToolMessage("assistant-middle", 4, undefined, sessionID),
        buildUserMessage("user-latest", "Current violet widget policy wins.", 5, sessionID),
        buildAssistantToolMessage("assistant-latest", 6, 9_000, sessionID),
    ]
    const handoff = [
        "## Decisions",
        "- The legacy widget policy is superseded by violet.",
        "## Files & Symbols",
        "- src/widget.ts and widget tests",
        "## Errors (verbatim)",
        "- (none)",
        "## What failed and why",
        "- The previous rule failed a widget test.",
        "## Constraints",
        "- Check widget tests carefully. Current violet widget policy wins.",
        "## Next step",
        "- Apply the current violet policy and verify widget tests.",
    ].join("\n")
    let liveCalls = 0
    let descriptionCalls = 0
    let descriptionResolved = false
    let signalLivePrompt!: () => void
    const livePromptStarted = new Promise<void>((resolve) => {
        signalLivePrompt = resolve
    })
    let releaseHandoff!: () => void
    const liveGate = new Promise<void>((resolve) => {
        releaseHandoff = resolve
    })
    let releaseDescription!: () => void
    const descriptionGate = new Promise<void>((resolve) => {
        releaseDescription = resolve
    })
    const client = {
        ...transformClient(10_000),
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "anthropic",
                            models: { "claude-test": { limit: { context: 10_000 } } },
                        },
                        {
                            id: "openai",
                            models: {
                                "gpt-6-luna": {
                                    limit: { context: 100_000 },
                                    variants: { high: {} },
                                },
                            },
                        },
                    ],
                },
            }),
        },
        session: {
            get: async () => ({ data: { parentID: null } }),
            create: async () => ({ data: { id: `scratch-${++liveCalls}` } }),
            prompt: async ({ body }: any) => {
                if ((body.parts[0].text as string).includes("Describe only archive")) {
                    assert.equal(body.variant, undefined)
                    descriptionCalls++
                    await descriptionGate
                    descriptionResolved = true
                    return {
                        data: {
                            parts: [
                                {
                                    type: "text",
                                    text: JSON.stringify({
                                        description:
                                            "Widget migration archive with legacy test evidence.",
                                    }),
                                },
                            ],
                        },
                    }
                }
                assert.equal(body.variant, "high")
                signalLivePrompt()
                await liveGate
                return { data: { parts: [{ type: "text", text: JSON.stringify({ handoff }) }] } }
            },
            delete: async () => ({ data: true }),
        },
    }
    const config = profileEscalationConfig(true, 45)
    config.compaction.summaryModel = "openai/gpt-6-luna"
    const runtime = createRuntimeState(client, new Logger(false))
    const output = { messages: structuredClone(messages) }
    let transformFinished = false
    const transform = transformHandler(
        client,
        runtime,
        config,
        directory,
    )({}, output).then(() => {
        transformFinished = true
    })
    try {
        await livePromptStarted
        assert.equal(
            transformFinished,
            false,
            "the next provider request must wait for the validated handoff",
        )
    } finally {
        releaseHandoff()
    }
    await transform
    assert.ok(liveCalls >= 1)
    assert.match(JSON.stringify(output.messages), /Secret legacy widget policy must apply/)
    assert.match(JSON.stringify(output.messages), /Current violet widget policy wins/)
    const catalog = await loadArchiveCatalog(directory, sessionID)
    assert.ok(catalog.checkpoint)
    assert.equal(catalog.retirementThrough, undefined)
    assert.equal(
        descriptionResolved,
        false,
        "the pre-provider hook must not wait for background descriptions",
    )
    releaseDescription()
    await waitFor(() => descriptionCalls > 0)
})

test("manual request during a busy turn waits until idle and runs once", async () => {
    const sessionID = `ses-idle-queue-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionID)
    const config = profileEscalationConfig(false, 10)
    const directory = mkdtempSync(join(tmpdir(), "better-compact-idle-queue-"))
    let status: "busy" | "idle" = "busy"
    const reports: any[] = []
    const client = {
        session: {
            status: async () => ({ data: { [sessionID]: { type: status } } }),
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async (request: any) => {
                reports.push(request)
                return { data: true }
            },
        },
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "anthropic",
                            models: { "claude-test": { limit: { context: 50_000 } } },
                        },
                    ],
                },
            }),
        },
        tui: { showToast: async () => ({}) },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const state = runtime.get(sessionID)
    state.modelContextLimit = 50_000
    const permissions = { global: undefined, agents: {} }
    const handler = createChatMessageHandler(
        client as any,
        runtime,
        logger,
        config,
        directory,
        permissions,
    )
    await handler(
        { sessionID, model: { providerID: "anthropic", modelID: "claude-test" } },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        contextLimit: 50_000,
                        chatVariant: "xhigh",
                        chatProviderID: "anthropic",
                        chatModelID: "claude-test",
                        compaction: { preset: "custom", custom: { targetPercent: 18 } },
                    },
                },
            ],
        },
    )
    assert.ok(state.boundary.queuedManual)
    assert.equal(state.boundary.queuedManual?.params?.variant, "xhigh")
    assert.equal(state.boundary.queuedManual?.compaction?.custom?.targetPercent, 18)
    assert.equal(state.boundary.activePlan, null)
    assert.equal(reports.length, 0)
    status = "idle"
    // Simulate an OpenCode restart before the idle event arrives: the queue
    // must survive on disk and initialize in the new plugin runtime.
    const resumed = createRuntimeState(client, logger)
    const event = createEventHandler(resumed, logger, client, config, directory, permissions)
    await event({ event: { type: "session.idle", properties: { sessionID } } })
    assert.equal(resumed.get(sessionID).boundary.queuedManual, undefined)
    assert.ok(resumed.get(sessionID).boundary.activePlan)
    assert.equal(reports.length, 1)
    assert.equal(reports[0].body.variant, "xhigh")
    assert.equal(resumed.get(sessionID).boundary.activePlan?.targetTokens, 9_000)
    assert.equal(resumed.get(sessionID).boundary.queuedManual, undefined)
    await event({ event: { type: "session.idle", properties: { sessionID } } })
    assert.equal(reports.length, 1)
})

test("the report no-reply prompt uses the observed chat variant without guessing from session.get", async () => {
    let prompt: any
    const client = {
        session: {
            get: async () => ({ data: { parentID: null } }),
            prompt: async (input: any) => {
                prompt = input
            },
        },
    }
    await sendIgnoredMessage(
        client,
        "ses-variant-change",
        "Better Compact complete",
        {
            providerId: "openai",
            modelId: "gpt-6-sol",
            variant: "xhigh",
            agent: "build",
        },
        new Logger(false),
    )
    assert.equal(prompt.body.noReply, true)
    assert.equal(prompt.body.variant, "xhigh")
    assert.equal(prompt.body.agent, "build")
})

test("slash command during a busy turn queues rather than compacting mid-loop", async () => {
    const sessionID = `ses-command-queue-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionID)
    const lastUser = messages.at(-1)!.info
    if (lastUser.role === "user") (lastUser as any).model.variant = "xhigh"
    const config = profileEscalationConfig(false, 10)
    const directory = mkdtempSync(join(tmpdir(), "better-compact-command-queue-"))
    let status: "busy" | "idle" = "busy"
    let reports = 0
    const client = {
        session: {
            status: async () => ({ data: { [sessionID]: { type: status } } }),
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async () => {
                reports++
                return { data: true }
            },
        },
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "anthropic",
                            models: { "claude-test": { limit: { context: 50_000 } } },
                        },
                    ],
                },
            }),
        },
        tui: { showToast: async () => ({}) },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    runtime.get(sessionID).modelContextLimit = 50_000
    const permissions = { global: undefined, agents: {} }
    const output = { parts: [{ type: "text", text: "command" }] }
    await createCommandExecuteHandler(
        client as any,
        runtime,
        logger,
        config,
        directory,
        permissions,
    )({ command: "better-compact", sessionID, arguments: "compress" }, output)
    assert.equal(output.parts.length, 0)
    assert.ok(runtime.get(sessionID).boundary.queuedManual)
    assert.equal(runtime.get(sessionID).boundary.activePlan, null)
    status = "idle"
    await createEventHandler(
        runtime,
        logger,
        client,
        config,
        directory,
        permissions,
    )({ event: { type: "session.idle", properties: { sessionID } } })
    assert.ok(runtime.get(sessionID).boundary.activePlan)
    assert.equal(reports, 0, "an unobserved variant must not be reset by a report prompt")
})

test("a queued manual compaction runs before the next user turn's provider request if idle was missed", async () => {
    const sessionID = `ses-request-queue-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionID)
    const config = profileEscalationConfig(false, 10)
    const directory = mkdtempSync(join(tmpdir(), "better-compact-request-queue-"))
    let reports = 0
    const client = {
        session: {
            status: async () => ({ data: { [sessionID]: { type: "busy" } } }),
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async () => {
                reports++
                return { data: true }
            },
        },
        provider: {
            list: async () => ({
                data: {
                    all: [
                        {
                            id: "anthropic",
                            models: { "claude-test": { limit: { context: 50_000 } } },
                        },
                    ],
                },
            }),
        },
        tui: { showToast: async () => ({}) },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const state = runtime.get(sessionID)
    state.modelContextLimit = 50_000
    const permissions = { global: undefined, agents: {} }
    await createChatMessageHandler(
        client as any,
        runtime,
        logger,
        config,
        directory,
        permissions,
    )(
        { sessionID, model: { providerID: "anthropic", modelID: "claude-test" } },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: { betterCompact: "run", contextLimit: 50_000 },
                },
            ],
        },
    )
    assert.ok(state.boundary.queuedManual)
    const output = {
        messages: [...messages, buildUserMessage("queued-next-user", "new turn", 99, sessionID)],
    }
    await createChatMessageTransformHandler(
        client as any,
        runtime,
        logger,
        config,
        permissions,
        directory,
    )({}, output)
    assert.equal(state.boundary.queuedManual, undefined)
    assert.ok(state.boundary.activePlan)
    assert.ok(output.messages.some((message) => message.info.id.startsWith("msg_better_compact_")))
    assert.equal(reports, 0, "pre-request execution must not append another user prompt")
})

test("a busy manual request waits for new tool output and runs before the same user turn continues", async () => {
    const sessionID = `ses-tool-queue-${Date.now()}`
    const messages = buildProfileEscalationConversation(sessionID)
    const directory = mkdtempSync(join(tmpdir(), "better-compact-tool-queue-"))
    const config = profileEscalationConfig(false, 10)
    const client = {
        ...transformClient(50_000),
        session: {
            status: async () => ({ data: { [sessionID]: { type: "busy" } } }),
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async () => {
                throw new Error("No nested prompt during an agent loop")
            },
        },
    }
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const permissions = { global: undefined, agents: {} }
    runtime.get(sessionID).modelContextLimit = 50_000
    await createChatMessageHandler(
        client,
        runtime,
        logger,
        config,
        directory,
        permissions,
    )(
        { sessionID, model: { providerID: "anthropic", modelID: "claude-test" } },
        {
            message: { agent: "assistant" },
            parts: [
                {
                    type: "text",
                    ignored: true,
                    metadata: { betterCompact: "run", contextLimit: 50_000 },
                },
            ],
        },
    )
    const state = runtime.get(sessionID)
    const handler = createChatMessageTransformHandler(
        client,
        runtime,
        logger,
        config,
        permissions,
        directory,
    )
    await handler({}, { messages: structuredClone(messages) })
    assert.ok(
        state.boundary.queuedManual,
        "same content must not start compaction in the queuing request",
    )
    assert.equal(state.boundary.activePlan, null)

    const progressed = [
        ...messages,
        buildAssistantToolMessage("assistant-tool-finished", 100, undefined, sessionID),
    ]
    const output = { messages: structuredClone(progressed) }
    await handler({}, output)
    assert.equal(state.boundary.queuedManual, undefined)
    assert.ok(state.boundary.activePlan)
    assert.ok(output.messages.some((message) => message.info.id.startsWith("msg_better_compact_")))
    assert.equal(getLastUserMessage(progressed)?.info.id, getLastUserMessage(messages)?.info.id)
    await handler({}, { messages: structuredClone(progressed) })
    assert.equal(state.boundary.queuedManual, undefined, "the same request must run once")
})

test("automatic hook uses exact model budgets and falls back for other models", async () => {
    for (const [tokens, model, expected] of [
        [180999, "claude-test", false],
        [181000, "claude-test", true],
        [181000, "other", false],
    ] as const) {
        const sessionID = `override-${model}-${tokens}-${process.pid}`
        const client = transformClient(262144, [])
        const runtime = createRuntimeState(client, new Logger(false))
        runtime.setModelLimit("anthropic", model, 262144)
        const config = buildConfig("allow")
        config.compaction.providers = {
            anthropic: {
                triggerTokens: 200000,
                models: { "claude-test": { triggerTokens: 181000, targetTokens: 90000 } },
            },
        }
        const messages = [
            buildUserMessage("user-1", "old request", 1, sessionID),
            buildReducibleOldToolMessage("assistant-1", 2, sessionID),
            buildUserMessage("user-2", "middle request", 3, sessionID),
            buildAssistantToolMessage("assistant-2", 4, tokens, sessionID),
            buildUserMessage("user-3", "latest request", 5, sessionID),
        ]
        for (const message of messages)
            if (message.info.role === "user") (message.info as any).model.modelID = model
        await finishAutoTurn(
            client,
            runtime,
            config,
            mkdtempSync(join(tmpdir(), "better-compact-override-")),
            messages,
            sessionID,
        )
        const plan = runtime.get(sessionID).boundary.activePlan
        assert.equal(Boolean(plan), expected)
        if (plan) assert.equal(plan.targetTokens, 90000)
    }
})

test("concurrent idle checks share one committed plan for subsequent transforms", async () => {
    const sessionID = `session-concurrent-${process.pid}-${Date.now()}`
    const toasts: unknown[] = []
    const client = transformClient(100_000, toasts)
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-concurrent-"))
    const handler = transformHandler(client, runtime, config, directory)
    const messages = [
        buildUserMessage("user-1", "old request", 1, sessionID),
        buildReducibleOldToolMessage("assistant-1", 2, sessionID),
        buildUserMessage("user-2", "middle request", 3, sessionID),
        buildAssistantToolMessage("assistant-2", 4, 90_000, sessionID),
        buildUserMessage("user-3", "latest request", 5, sessionID),
    ]
    const first = { messages: structuredClone(messages) }
    const second = { messages: structuredClone(messages) }

    await Promise.all([
        finishAutoTurn(client, runtime, config, directory, messages, sessionID),
        finishAutoTurn(client, runtime, config, directory, messages, sessionID),
    ])
    await Promise.all([handler({}, first), handler({}, second)])

    assert.ok(runtime.get(sessionID).boundary.activePlan)
    assert.ok(first.messages.some((message) => message.info.id.startsWith("msg_better_compact_")))
    assert.ok(second.messages.some((message) => message.info.id.startsWith("msg_better_compact_")))
    assert.equal(toasts.length, 1)
})

test("message transform resolves the active model limit before automatic planning", async () => {
    const client = {
        session: { get: async () => ({ data: { parentID: null } }) },
        provider: {
            list: async () => [
                {
                    id: "anthropic",
                    models: {
                        "claude-test": { limit: { context: 1_000_000 } },
                        "claude-small": { limit: { context: 200_000 } },
                    },
                },
            ],
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const config = buildConfig("allow")
    config.compaction.automatic = false
    const handler = createChatMessageTransformHandler(
        client as any,
        runtime,
        new Logger(false),
        config,
        { global: undefined, agents: {} },
    )
    const largeModel = buildUserMessage("user-large", "first", 1)
    await handler({}, { messages: [largeModel] })
    assert.equal(runtime.get("session-1").modelContextLimit, 1_000_000)

    const smallModel = buildUserMessage("user-small", "second", 2)
    if (smallModel.info.role === "user") (smallModel.info as any).model.modelID = "claude-small"
    await handler({}, { messages: [smallModel] })

    assert.equal(runtime.get("session-1").modelContextLimit, 200_000)
})

test("automatic compaction replaces an active plan after another over-trigger turn ends", async () => {
    const sessionID = `session-replan-${process.pid}-${Date.now()}`
    const client = transformClient(100_000)
    const logger = new Logger(false)
    const runtime = createRuntimeState(client, logger)
    const state = runtime.get(sessionID)
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-replan-"))
    const initial = [
        buildUserMessage("user-1", "old request", 1, sessionID),
        buildReducibleOldToolMessage("assistant-1", 2, sessionID),
        buildUserMessage("user-2", "middle request", 3, sessionID),
        buildAssistantToolMessage("assistant-2", 4, 90_000, sessionID),
        buildUserMessage("user-3", "latest request", 5, sessionID),
    ]
    await finishAutoTurn(client, runtime, config, directory, initial, sessionID)
    const firstRangeHash = state.boundary.activePlan?.rangeHash
    assert.ok(firstRangeHash)
    assert.equal(runtime.activeCompaction(sessionID), undefined)

    const grown = [
        buildUserMessage("user-1", "old request", 1, sessionID),
        buildReducibleOldToolMessage("assistant-1", 2, sessionID),
        buildUserMessage("user-2", "middle request", 3, sessionID),
        buildAssistantToolMessage("assistant-2", 4, undefined, sessionID),
        buildUserMessage("user-3", "next request", 5, sessionID),
        buildAssistantToolMessage("assistant-3", 6, undefined, sessionID),
        buildUserMessage("user-4", "more work", 7, sessionID),
        buildAssistantToolMessage("assistant-4", 8, 95_000, sessionID),
        buildUserMessage("user-5", "latest request", 9, sessionID),
    ]
    await finishAutoTurn(client, runtime, config, directory, grown, sessionID)

    assert.notEqual(state.boundary.activePlan?.rangeHash, firstRangeHash)
    assert.equal(state.boundary.activePlan?.rawTailStartMessageId, "user-4")
})

test("a failed transcript write stops an overflowing provider request", async () => {
    const sessionId = `ses-transform-guard-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionId), 9_000)
    const client = transformClient(10_000)
    const runtime = createRuntimeState(client, new Logger(false))
    const brokenDirectory = join(
        mkdtempSync(join(tmpdir(), "better-compact-broken-")),
        "not-a-directory",
    )
    writeFileSync(brokenDirectory, "regular file blocking mkdir")
    const config = buildConfig("allow")
    const handler = transformHandler(client, runtime, config, brokenDirectory)
    const output = { messages }

    await finishAutoTurn(client, runtime, config, brokenDirectory, messages, sessionId)
    await assert.rejects(handler({}, output), /provider request was stopped/)

    assert.ok(!messages.some((item) => item.info.id.startsWith("msg_better_compact_context_")))
    assert.ok(messages.some((item) => item.parts.some((part) => part.type === "tool")))
})

test("compaction failure diagnostics never echo provider prompts or private paths", async () => {
    const { safeCompactionFailure } = await import("../lib/hooks")
    assert.equal(
        safeCompactionFailure(
            new Error("Provider rejected: private user correction and key=secret"),
        ),
        "unexpected_error",
    )
    assert.equal(safeCompactionFailure(new Error("archive_io_error")), "archive_io_error")
    assert.equal(
        safeCompactionFailure("/home/user/private/.opencode/catalog.json"),
        "unexpected_error",
    )
})

test("idle automatic compaction clears a stale snapshot and rebuilds a fresh plan", async () => {
    const sessionId = `ses-transform-stale-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionId), 9_000)
    const logger = new Logger(false)
    const seeded = createSessionState(sessionId)
    seeded.boundary.activePlan = {
        sessionId,
        rangeHash: "deadbeefdeadbeef",
        contextLimit: 10_000,
        rawTailStartMessageId: "user-2",
        transcriptRelativePath: ".opencode/better-compact/sessions/stale/stale.md",
        beforeTokens: 9_000,
        afterPruneTokens: 1_000,
        overheadTokens: 0,
        triggerTokens: 8_500,
        targetTokens: 3_000,
        requiresCustomCompaction: false,
        stages: [],
        createdAt: Date.now(),
    }
    await saveSessionState(seeded, logger)
    const client = transformClient(10_000)
    const runtime = createRuntimeState(client, logger)
    const config = buildConfig("allow")
    const directory = mkdtempSync(join(tmpdir(), "better-compact-transform-stale-"))
    const handler = transformHandler(client, runtime, config, directory)
    const output = { messages }

    await finishAutoTurn(client, runtime, config, directory, messages, sessionId)
    await handler({}, output)

    const state = runtime.get(sessionId)
    assert.ok(state.boundary.activePlan)
    assert.notEqual(state.boundary.activePlan?.rangeHash, "deadbeefdeadbeef")
    assert.ok(messages.some((item) => item.info.id.startsWith("msg_better_compact_context_")))
})

test("command execute exits after effective permission resolves to deny", async () => {
    let sessionMessagesCalls = 0
    const output = { parts: [] as any[] }
    const client = {
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => {
                sessionMessagesCalls += 1
                return { data: [] }
            },
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const handler = createCommandExecuteHandler(
        client as any,
        runtime,
        new Logger(false),
        buildConfig("deny"),
        "/tmp",
        { global: undefined, agents: {} },
    )

    await handler(
        { command: "better-compact", sessionID: "session-1", arguments: "context" },
        output,
    )

    assert.equal(sessionMessagesCalls, 1)
    assert.deepEqual(output.parts, [])
})

test("better-compact stores virtual plan and reports progress without native summarize", async () => {
    const messages = [
        buildUserMessage("user-1", "old user request", 1),
        buildAssistantToolMessage("assistant-1", 2),
        buildUserMessage("user-2", "middle user request", 3),
        buildMessage("assistant-2", "assistant", "middle assistant response"),
        buildUserMessage("user-3", "latest user request", 5),
    ]
    if (messages.at(-1)?.info.role === "user")
        (messages.at(-1)!.info as any).model.variant = "xhigh"
    const prompts: any[] = []
    const notifications: any[] = []
    let summarizeCalls = 0
    const output = { parts: [{ type: "text", text: "/better-compact" }] as any[] }
    const client = {
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async (input: any) => {
                prompts.push(input)
                return { data: true }
            },
            summarize: async () => {
                summarizeCalls += 1
                throw new Error("native compaction should not be called")
            },
        },
        tui: {
            showToast: async (input: any) => {
                notifications.push(input)
            },
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get("session-1")
    const handler = createCommandExecuteHandler(
        client as any,
        runtime,
        new Logger(false),
        buildConfig("allow"),
        mkdtempSync(join(tmpdir(), "better-compact-command-")),
        { global: undefined, agents: {} },
    )

    await handler({ command: "better-compact", sessionID: "session-1", arguments: "" }, output)
    await waitFor(() => state.boundary.job?.status === "completed")

    assert.equal(summarizeCalls, 0)
    assert.equal(output.parts.length, 0)
    assert.ok(state.boundary.activePlan)
    assert.equal(state.boundary.activePlan?.sessionId, "session-1")
    assert.equal(prompts.length, 0, "unknown variant must not create a new session message")
    assert.match(
        notifications.map((notification) => notification.body.message).join("\n"),
        /Better Compact Complete/,
    )
    assert.equal(state.boundary.job?.percent, 100)
    assert.equal(state.boundary.job?.counters.contextLimit, 200_000)
    assert.ok((state.boundary.job?.counters.beforeTokens ?? 0) > 0)
    assert.ok((state.boundary.job?.counters.currentTokens ?? 0) > 0)
    assert.ok(state.boundary.job?.logs.some((line) => line.includes("Transcript written")))
})

test("concurrent better-compact runs for a session are rejected while one is in flight", async () => {
    const messages = [
        buildUserMessage("user-1", "old user request", 1),
        buildAssistantToolMessage("assistant-1", 2),
        buildUserMessage("user-2", "middle user request", 3),
        buildMessage("assistant-2", "assistant", "middle assistant response"),
        buildUserMessage("user-3", "latest user request", 5),
    ]
    const notifications: any[] = []
    const releases: Array<() => void> = []
    const client = {
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async () => {
                throw new Error("unknown variant must not send a prompt")
            },
        },
        tui: {
            showToast: async (input: any) => {
                notifications.push(input)
                if (input.body.message.includes("Better Compact Complete"))
                    await new Promise<void>((resolve) => releases.push(resolve))
            },
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get("session-1")
    const handler = createCommandExecuteHandler(
        client as any,
        runtime,
        new Logger(false),
        buildConfig("allow"),
        mkdtempSync(join(tmpdir(), "better-compact-concurrent-manual-")),
        { global: undefined, agents: {} },
    )

    // The first run blocks in its deferred final-report toast; the second
    // must be turned away by the single-flight guard without starting a job.
    const first = handler(
        { command: "better-compact", sessionID: "session-1", arguments: "" },
        { parts: [] as any[] },
    )
    await first
    await waitFor(() => notifications.length === 1)
    const second = handler(
        { command: "better-compact", sessionID: "session-1", arguments: "" },
        { parts: [] as any[] },
    )
    await waitFor(() => notifications.length === 2)
    releases.splice(0).forEach((release) => release())
    await second
    await waitFor(() => state.boundary.job?.status === "completed")
    releases.splice(0).forEach((release) => release())
    await waitFor(() => runtime.activeCompaction("session-1") === undefined)

    const texts = notifications.map((notification) => notification.body.message)
    assert.equal(texts.filter((text) => /already running/.test(text)).length, 1)
    assert.equal(texts.filter((text) => /Better Compact Complete/.test(text)).length, 1)
})

test("chat message sentinel runs better-compact as no-reply TUI action", async () => {
    const messages = [
        buildUserMessage("user-1", "old user request", 1),
        buildAssistantToolMessage("assistant-1", 2),
        buildUserMessage("user-2", "middle user request", 3),
        buildMessage("assistant-2", "assistant", "middle assistant response"),
        buildUserMessage("user-3", "latest user request", 5),
    ]
    const prompts: any[] = []
    const client = {
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: messages }),
            prompt: async (input: any) => {
                prompts.push(input)
                return { data: true }
            },
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get("session-1")
    const handler = createChatMessageHandler(
        client as any,
        runtime,
        new Logger(false),
        buildConfig("allow"),
        mkdtempSync(join(tmpdir(), "better-compact-sentinel-")),
        { global: undefined, agents: {} },
    )

    await handler(
        {
            sessionID: "session-1",
        },
        {
            message: {
                agent: "assistant",
                model: { providerID: "anthropic", modelID: "claude-test" },
                variant: "high",
            },
            parts: [
                {
                    type: "text",
                    text: "Better Compact requested.",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        jobId: "bc_tuitest",
                        jobStartedAt: 123_456,
                        summaryVariant: "high",
                        chatVariant: "xhigh",
                        chatProviderID: "anthropic",
                        chatModelID: "claude-test",
                        summaryProviderID: "anthropic",
                        summaryModelID: "claude-test",
                        contextLimit: 1_000_000,
                        currentTokens: 857_703,
                    },
                },
            ],
        },
    )
    await waitFor(() => state.boundary.job?.status === "completed")

    assert.ok(state.boundary.activePlan)
    assert.equal(state.boundary.activePlan?.sessionId, "session-1")
    assert.ok(prompts.length >= 1)
    assert.ok(prompts.every((prompt) => prompt.body.variant === "xhigh"))
    assert.equal(
        prompts.every(
            (prompt) =>
                prompt.body.model.providerID === "anthropic" &&
                prompt.body.model.modelID === "claude-test",
        ),
        true,
    )
    assert.match(
        prompts.map((prompt) => prompt.body.parts[0].text).join("\n"),
        /Better Compact Complete/,
    )
    assert.equal(state.boundary.job?.percent, 100)
    assert.equal(state.boundary.job?.id, "bc_tuitest")
    assert.equal(state.boundary.job?.startedAt, 123_456)
    assert.equal(state.boundary.job?.counters.contextLimit, 1_000_000)
    assert.equal(state.boundary.job?.counters.beforeTokens, 857_703)
    assert.ok((state.boundary.job?.counters.currentTokens ?? 0) > 0)
    assert.ok(
        state.boundary.job?.stages.some(
            (stage) => stage.id === "report" && stage.status === "completed",
        ),
    )
})

test("denied TUI compaction persists a correlated failed job", async () => {
    const sessionID = `session-denied-${Date.now()}`
    const client = {
        session: {
            get: async () => ({ data: { parentID: null } }),
            messages: async () => ({ data: [] }),
        },
    }
    const runtime = createRuntimeState(client, new Logger(false))
    const state = runtime.get(sessionID)
    const handler = createChatMessageHandler(
        client as any,
        runtime,
        new Logger(false),
        buildConfig("deny"),
        mkdtempSync(join(tmpdir(), "better-compact-denied-")),
        { global: undefined, agents: {} },
    )

    await handler(
        { sessionID },
        {
            message: {},
            parts: [
                {
                    type: "text",
                    text: "Better Compact requested.",
                    ignored: true,
                    metadata: {
                        betterCompact: "run",
                        jobId: "bc_denied",
                        jobStartedAt: 123_456,
                        contextLimit: 272_000,
                        currentTokens: 200_000,
                        targetTokens: 95_200,
                    },
                },
            ],
        },
    )

    assert.equal(state.boundary.job?.id, "bc_denied")
    assert.equal(state.boundary.job?.status, "failed")
    assert.match(state.boundary.job?.error ?? "", /denied/i)
    assert.equal(state.boundary.job?.counters.contextLimit, 272_000)
    assert.equal(state.boundary.job?.counters.targetTokens, 95_200)
})

test("text complete strips hallucinated metadata tags", async () => {
    const output = { text: "alpha <dcp>beta</dcp> omega" }
    const handler = createTextCompleteHandler()

    await handler({ sessionID: "session-1", messageID: "message-1", partID: "part-1" }, output)

    assert.equal(output.text, "alpha  omega")
})

function buildForkFixture(sessionID: string): { messages: WithParts[]; prefix: WithParts[] } {
    const messages = [
        buildUserMessage(`${sessionID}-user-1`, `shared fork prefix for ${sessionID}`, 1, sessionID),
        buildMessage(`${sessionID}-assistant-1`, "assistant", `shared fork answer for ${sessionID}`),
        buildUserMessage(`${sessionID}-user-2`, "fork raw tail", 3, sessionID),
        buildUserMessage(`${sessionID}-user-3`, "fork latest", 5, sessionID),
    ]
    messages[1].info.sessionID = sessionID
    for (const part of messages[1].parts) part.sessionID = sessionID
    return { messages, prefix: messages.slice(0, 2) }
}

async function persistForkSourcePlan(directory: string, prefix: WithParts[]): Promise<string> {
    const { boundaryRangeHash } = await import("../lib/boundary/fingerprint")
    const { mkdirSync } = await import("node:fs")
    const transcriptRelativePath = ".opencode/better-compact/sessions/fork-source/plan.md"
    mkdirSync(join(directory, ".opencode/better-compact/sessions/fork-source"), { recursive: true })
    writeFileSync(join(directory, transcriptRelativePath), "source transcript")
    const source = createSessionState(
        `fork-source-${Date.now()}-${Math.random().toString(16).slice(2)}`,
    )
    source.boundary.activePlan = {
        sessionId: source.sessionId!,
        rangeHash: "0123456789abcdef",
        contextLimit: 100_000,
        rawTailStartMessageId: "source-raw-tail",
        prefixFingerprint: boundaryRangeHash(prefix),
        compactedMessageCount: prefix.length,
        transcriptRelativePath,
        beforeTokens: 90_000,
        afterPruneTokens: 20_000,
        overheadTokens: 0,
        triggerTokens: 85_000,
        targetTokens: 35_000,
        prefixSummaryAllowed: false,
        collapsePercent: 25,
        requiresCustomCompaction: false,
        stages: [],
        createdAt: 1,
    }
    await saveSessionState(source, new Logger(false))
    const owned = structuredClone(prefix)
    for (const message of owned) {
        message.info.sessionID = source.sessionId!
        for (const part of message.parts) part.sessionID = source.sessionId!
    }
    await archiveBoundaryDelta({
        directory,
        sessionId: source.sessionId!,
        plan: {
            sessionId: source.sessionId!,
            rangeHash: "fork-source-range",
            transcript: { relativePath: transcriptRelativePath, content: "", messageIds: owned.map((m) => m.info.id) },
        } as any,
        originalMessages: owned,
    })
    return source.sessionId!
}

test("fork plan inheritance is skipped when compress permission is deny", async () => {
    const directory = mkdtempSync(join(tmpdir(), "better-compact-inherit-deny-"))
    const stamp = `${process.pid}-${Date.now()}`
    const allowSession = `fork-child-allow-${stamp}`
    const denySession = `fork-child-deny-${stamp}`
    const allowFixture = buildForkFixture(allowSession)
    const denyFixture = buildForkFixture(denySession)
    const allowSourceId = await persistForkSourcePlan(directory, allowFixture.prefix)
    const denySourceId = await persistForkSourcePlan(directory, denyFixture.prefix)

    // Same fixtures, permission allow: the plan is inherited.
    const allowClient = {
        ...transformClient(100_000),
        session: {
            get: async ({ path }: { path: { id: string } }) => ({
                data: {
                    parentID: null,
                    title: path.id === allowSession ? "Tracked project (fork #1)" :
                        path.id === allowSourceId ? "Tracked project" : "Independent session",
                },
            }),
        },
    }
    const allowRuntime = createRuntimeState(allowClient, new Logger(false))
    await transformHandler(
        allowClient,
        allowRuntime,
        buildConfig("allow"),
        directory,
    )({}, { messages: allowFixture.messages })
    assert.ok(
        (await loadArchiveCatalog(directory, allowSession)).inheritedLinks?.some(
            (link) => link.ownerSessionId === allowSourceId,
        ),
        "a content-matched fork should inherit its owner's read-only catalog link",
    )

    // Identical content in a separately created session is not a fork.
    const unrelatedSession = `independent-${stamp}`
    const unrelatedMessages = structuredClone(allowFixture.messages)
    for (const message of unrelatedMessages) {
        message.info.sessionID = unrelatedSession
        for (const part of message.parts) part.sessionID = unrelatedSession
    }
    await transformHandler(
        allowClient,
        createRuntimeState(allowClient, new Logger(false)),
        buildConfig("allow"),
        directory,
    )({}, { messages: unrelatedMessages })
    assert.deepEqual((await loadArchiveCatalog(directory, unrelatedSession)).inheritedLinks ?? [], [])

    // Same fixtures, permission deny: inheritance must not even happen.
    const denyClient = {
        ...transformClient(100_000),
        session: {
            get: async ({ path }: { path: { id: string } }) => ({
                data: {
                    parentID: null,
                    title: path.id === denySession ? "Denied project (fork #1)" :
                        path.id === denySourceId ? "Denied project" : "Independent session",
                },
            }),
        },
    }
    const denyRuntime = createRuntimeState(denyClient, new Logger(false))
    await transformHandler(
        denyClient,
        denyRuntime,
        buildConfig("deny"),
        directory,
    )({}, { messages: denyFixture.messages })
    assert.equal(denyRuntime.get(denySession).boundary.activePlan, null)
    assert.deepEqual((await loadArchiveCatalog(directory, denySession)).inheritedLinks ?? [], [])
})

test("a waiting idle check is not rejected when the active compaction fails", async () => {
    const sessionId = `ses-loser-shield-${Date.now()}`
    const messages = withProviderUsage(buildOverTriggerConversation(sessionId), 9_000)
    const toasts: unknown[] = []
    const client = transformClient(10_000, toasts)
    const runtime = createRuntimeState(client, new Logger(false))
    let rejectWinner!: (error: Error) => void
    runtime.startCompaction(
        sessionId,
        () =>
            new Promise((_, reject) => {
                rejectWinner = reject
            }),
    )
    const before = JSON.stringify(messages)
    const waiting = finishAutoTurn(
        client,
        runtime,
        buildConfig("allow"),
        mkdtempSync(join(tmpdir(), "better-compact-loser-shield-")),
        messages,
        sessionId,
    )
    await new Promise((resolve) => setTimeout(resolve, 25))
    rejectWinner(new Error("winner exploded"))
    await waiting

    // The waiting transform degrades quietly: no rejection, no failure toast
    // of its own, request continues unpruned.
    assert.equal(toasts.length, 0)
    assert.equal(JSON.stringify(messages), before)
    await waitFor(() => runtime.activeCompaction(sessionId) === undefined)
})

// Anthropic rejects a request whose latest assistant message carries thinking
// blocks that differ from what it issued, and the signature lives in the
// reasoning part's metadata. OpenCode already applies its own differentModel
// policy after this hook (it converts reasoning to plain text rather than
// emitting an unsigned thinking block), so the plugin must leave reasoning
// metadata alone — stripping it here produced a signature-less thinking block
// and a 400.
test("the transform preserves reasoning signatures even when the model differs", async () => {
    const logger = new Logger(false)
    const config = buildConfig("deny")
    const client = { session: { get: async () => ({}) }, provider: { list: async () => [] } }
    const runtime = createRuntimeState(client as any, logger)
    const handler = createChatMessageTransformHandler(client as any, runtime, logger, config, {
        global: undefined,
        agents: {},
    })
    const signature = "ErUBCkYIBRgCKkA0signature"
    const assistant: WithParts = {
        info: {
            id: "assistant-reasoning",
            role: "assistant",
            sessionID: "session-1",
            agent: "assistant",
            providerID: "anthropic",
            modelID: "claude-opus-4-8",
            time: { created: 2 },
        } as WithParts["info"],
        parts: [
            {
                id: "assistant-reasoning-part",
                messageID: "assistant-reasoning",
                sessionID: "session-1",
                type: "reasoning",
                text: "thinking through it",
                metadata: { anthropic: { signature } },
            } as WithParts["parts"][number],
        ],
    }
    // The newest user message names a different model, which is exactly what
    // happens after switching models mid-session.
    const output = {
        messages: [assistant, buildUserMessage("user-2", "carry on", 3)],
    }

    await handler({}, output)

    const reasoning = output.messages[0]?.parts[0] as { type: string; metadata?: unknown }
    assert.equal(reasoning.type, "reasoning", "the part must stay a reasoning part")
    assert.deepEqual(
        reasoning.metadata,
        { anthropic: { signature } },
        "the thinking signature must survive the transform verbatim",
    )
})
