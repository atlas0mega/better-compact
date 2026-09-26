/** @jsxImportSource @opentui/solid */

import type { TuiPluginModule } from "@opencode-ai/plugin/tui"
import { registerCommands } from "./lib/tui/commands"
import {
    activeSessionID,
    currentContextUsage,
    loadBoundaryJob,
    loadConfig,
    resolveSummaryVariant,
} from "./lib/tui/data"
import { resolveCompactionProfile } from "./lib/config"
import { createBoundaryJob } from "./lib/boundary/progress"
import { openPanelModal, openProgressModal, showError } from "./lib/tui/modals"

const tui: TuiPluginModule["tui"] = async (api) => {
    const config = loadConfig(api)
    if (!config.enabled || !config.commands.enabled) return

    const runCommand = {
        title: "Better Compact",
        name: "better-compact.run",
        description: "Run Better Compact staged pruning",
        slashName: "better-compact",
        run: async () => {
            const sessionID = activeSessionID(api)
            if (!sessionID) {
                api.ui.toast({
                    title: "Better Compact",
                    message: "Open a session first.",
                    variant: "warning",
                })
                return
            }
            const currentConfig = loadConfig(api)
            const settings = currentConfig.compaction
            const profile = resolveCompactionProfile(currentConfig, settings)
            const usage = currentContextUsage(api, sessionID)
            const agent = api.state.session.get(sessionID)?.agent
            const summaryVariant =
                settings.summaryEffort === "inherit"
                    ? usage.variant
                    : resolveSummaryVariant(api, sessionID, settings.summaryEffort)
            const percent = usage.limit > 0 ? Math.round((usage.tokens / usage.limit) * 100) : 0
            const run = async () => {
                const existingJob = await loadBoundaryJob(sessionID)
                if (
                    existingJob?.status === "running" &&
                    Date.now() - existingJob.updatedAt < 5 * 60 * 1000
                ) {
                    showError(
                        api,
                        "Better Compact",
                        "Compaction is already running for this session.",
                    )
                    return
                }
                const initialJob = createBoundaryJob({
                    sessionId: sessionID,
                    counters: {
                        beforeTokens: usage.tokens,
                        currentTokens: usage.tokens,
                        targetTokens: Math.round((usage.limit * profile.targetPercent) / 100),
                        contextLimit: usage.limit,
                        stageClearedTokens: 0,
                        clearedTokens: 0,
                    },
                })
                openProgressModal(api, config, initialJob)
                try {
                    await api.client.session.prompt({
                        sessionID,
                        agent,
                        model:
                            usage.providerID && usage.modelID
                                ? { providerID: usage.providerID, modelID: usage.modelID }
                                : undefined,
                        // This no-reply prompt belongs to the chat session.
                        // The summary model's effort is scratch-only.
                        variant: usage.variant,
                        // Busy prompts join the host's existing run; idle
                        // prompts request compaction without starting a turn.
                        noReply: !["busy", "retry"].includes(
                            api.state.session.status(sessionID)?.type ?? "idle",
                        ),
                        parts: [
                            {
                                type: "text",
                                text: "Better Compact requested.",
                                ignored: true,
                                metadata: {
                                    betterCompact: "run",
                                    compaction: settings,
                                    summaryVariant,
                                    chatVariant: usage.variant,
                                    chatProviderID: usage.providerID,
                                    chatModelID: usage.modelID,
                                    summaryProviderID: usage.providerID,
                                    summaryModelID: usage.modelID,
                                    jobId: initialJob.id,
                                    jobStartedAt: initialJob.startedAt,
                                    contextLimit: usage.limit,
                                    currentTokens: usage.tokens,
                                    targetTokens: initialJob.counters.targetTokens,
                                },
                            },
                        ],
                    })
                } catch (error) {
                    showError(api, "Better Compact", error)
                }
            }
            if (usage.limit > 0 && percent < profile.triggerPercent) {
                api.ui.dialog.replace(() => (
                    <api.ui.DialogConfirm
                        title="Compact anyway?"
                        message={`Current context is about ${percent}% used, below the ${profile.triggerPercent}% ${profile.preset} trigger. Target is ${profile.targetPercent}%. Continue?`}
                        onConfirm={() => {
                            // DialogConfirm clears after this callback; defer replacing it with progress.
                            queueMicrotask(() => void run())
                        }}
                    />
                ))
                api.ui.dialog.setSize("medium")
                return
            }
            await run()
        },
    }

    registerCommands(api, [
        runCommand,
        {
            title: "Better Compact Settings",
            name: "better-compact.panel",
            description: "Open Better Compact settings",
            slashName: "better-compact-settings",
            run: () => openPanelModal(api, loadConfig(api)),
        },
    ])
    // A higher-priority layer wins the built-in session.compact dispatch and
    // reachable slash entry; OpenCode's native action remains accessible only
    // through its API, not an accidental TUI /compact or /summarize.
    registerCommands(
        api,
        [
            {
                ...runCommand,
                title: "Better Compact (instead of native /compact)",
                name: "session.compact",
                slashName: "compact",
                slashAliases: ["summarize"],
            },
        ],
        1_000,
    )
}

export default {
    id: "better-compact",
    tui,
} satisfies TuiPluginModule
