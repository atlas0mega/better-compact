import { Plugin } from "@opencode/plugin"
import { compactV2Context, lastProviderTokens } from "./context"
import { loadV2Config } from "./config"
import { expireV2Archives } from "./archive"
import { registerV2Recall } from "./recall"
import { registerV2RPC } from "./rpc"
import { compactV2Native, hasOpaqueProviderState } from "./native"
import type { PluginConfig } from "./config"

function safeFailureCode(error: unknown): string {
    const text = error instanceof Error ? error.message : ""
    return ["Unvalidated archive retirement boundary", "Unvalidated checkpoint or description",
        "Missing archive for checkpoint", "Conflicting checkpoint for this range",
        "Conflicting historical handoff identity", "Modified archive", "Invalid or foreign V2 archive catalog",
        "Duplicate V2 native message identity", "V2 catalog is busy", "V2 system-update messages"].find((code) =>
        text.includes(code)) ?? (error instanceof Error ? error.name : "unknown")
}

function safeFailureSite(error: unknown): string {
    if (!(error instanceof Error)) return "unknown"
    const frames = (error.stack ?? "").split("\n").slice(1)
    for (const frame of frames) {
        const site = /(?:packages\/core\/(?:src|dist)|packages\/opencode\/lib\/v2)\/([a-zA-Z0-9._/-]+):(\d+)(?::\d+)?/.exec(frame)
        if (site) return `${site[1]}:${site[2]}`
    }
    return "external"
}

/** V2-only server plugin; every request is validated before the provider sees it. */
export default Plugin.define({
    id: "better-compact",
    async setup(ctx) {
        const initial = await loadV2Config(ctx.location.directory)
        if (!initial.enabled) return
        await registerV2Recall(ctx)
        await registerV2RPC(ctx)
        const summaryFor = async (modelRef: { providerID: string; id: string; variant?: string }, config: PluginConfig) => {
            const models = (await ctx.model.list()).data
            const configured = config.compaction.summaryModel
            const slash = configured ? configured.indexOf("/") : -1
            const summaryRef = configured && slash > 0
                ? { providerID: configured.slice(0, slash), id: configured.slice(slash + 1) }
                : { providerID: modelRef.providerID, id: modelRef.id }
            const summaryModel = models.find((candidate) =>
                candidate.providerID === summaryRef.providerID && candidate.id === summaryRef.id)
            const requestedVariant = config.compaction.summaryEffort === "inherit"
                ? modelRef.variant : config.compaction.summaryEffort === "off"
                    ? null : config.compaction.summaryEffort
            if (!summaryModel || requestedVariant && !summaryModel.variants.some((variant) => variant.id === requestedVariant))
                return undefined
            const summaryVariant = requestedVariant ?? (config.compaction.summaryEffort === "inherit" &&
                summaryModel.variants.some((variant) => variant.id === "high") ? "high" : null)
            return {
                modelContextLimit: summaryModel.limit.context,
                summaryVariant,
                generate: async (prompt: string, variant: string | undefined) =>
                    (await ctx.generate.text({ prompt, model: { ...summaryRef, variant } }, {
                        signal: AbortSignal.timeout(120_000),
                    })).text,
            }
        }
        await ctx.session.hook("context", async (event) => {
            let phase = "session"
            let providerTokens = 0
            let contextLimit = 0
            let summaryAvailable = false
            try {
                const session = await ctx.session.get({ sessionID: event.sessionID })
                phase = "config"
                const config = await loadV2Config(session.location.directory, event.model)
                phase = "housekeeping"
                await expireV2Archives(session.location.directory, event.sessionID)
                if (session.parentID && !config.experimental.allowSubAgents) return
                phase = "model"
                const model = (await ctx.model.list()).data.find((candidate) =>
                    candidate.providerID === event.model.providerID && candidate.id === event.model.id)
                contextLimit = model?.limit.context ?? 0
                phase = "history"
                const history = await ctx.session.context({ sessionID: event.sessionID })
                providerTokens = lastProviderTokens(history) ?? 0
                phase = "summary-model"
                const summary = await summaryFor(event.model, config)
                summaryAvailable = !!summary
                phase = "transform"
                const outcome = await compactV2Context(event, {
                    config,
                    projectRoot: session.location.directory,
                    contextLimit: model?.limit.context,
                    providerReportedTokens: providerTokens || undefined,
                    ...(summary ? { summary } : {}),
                    trace: (current) => { phase = current },
                })
                phase = "status-write"
                await ctx.storage.set(`status/${event.sessionID}`, {
                    status: outcome.status,
                    reason: outcome.reason ?? "none",
                    systemMessageCount: event.messages.filter((message) => message.role === "system").length,
                    providerTokens,
                    contextLimit,
                    summaryAvailable,
                    errorPhase: "none",
                    beforeTokens: outcome.beforeTokens ?? 0,
                    afterTokens: outcome.afterTokens ?? outcome.beforeTokens ?? 0,
                    candidateTokens: outcome.candidateTokens ?? outcome.afterTokens ?? outcome.beforeTokens ?? 0,
                    validationFailure: outcome.validationFailure ?? "none",
                    selectedStrategy: outcome.selectedStrategy ?? "none",
                    handoffVisible: outcome.handoffVisible === true,
                    triggerTokens: outcome.triggerTokens ?? 0,
                    targetTokens: outcome.targetTokens ?? 0,
                    at: Date.now(),
                })
            } catch (error) {
                // A failed optional transform must not silently remove any
                // original request content. No prompt/raw exception details in logs.
                await ctx.storage.set(`status/${event.sessionID}`, {
                    status: "error", reason: safeFailureCode(error), errorPhase: phase,
                    errorSite: safeFailureSite(error),
                    selectedStrategy: "none", handoffVisible: false, summaryAvailable,
                    systemMessageCount: event.messages.filter((message) => message.role === "system").length,
                    providerTokens, contextLimit, beforeTokens: 0, afterTokens: 0,
                    candidateTokens: 0, triggerTokens: 0, targetTokens: 0,
                    validationFailure: "none", at: Date.now(),
                }).catch(() => undefined)
                console.warn("Better Compact V2 context transform declined", {
                    error: safeFailureCode(error),
                    phase,
                    site: safeFailureSite(error),
                })
            }
        })
        await ctx.session.hook("compaction", async (event) => {
            const session = await ctx.session.get({ sessionID: event.sessionID })
            const config = await loadV2Config(session.location.directory, event.model)
            if (!config.enabled || session.parentID && !config.experimental.allowSubAgents) return
            if (config.compress.permission !== "allow") throw new Error("Better Compact V2 compaction is not allowed by plugin settings")
            if (hasOpaqueProviderState(event.messages)) return
            // Native compaction may be configured on the provider rather than
            // the model. Model listings do not include provider-level settings.
            const provider = await ctx.provider.get({ providerID: event.model.providerID }).catch(() => undefined)
            if (!provider || provider.data.settings?.compaction?.type === "native") return
            const model = (await ctx.model.list()).data.find((candidate) =>
                candidate.providerID === event.model.providerID && candidate.id === event.model.id)
            // Keep the host's native encrypted provider checkpoint path intact.
            if (model?.settings?.compaction?.type === "native") return
            const summary = await summaryFor(event.model, config)
            if (!summary) throw new Error("Configured V2 summary model or effort is unavailable")
            event.result = { summary: await compactV2Native({
                root: session.location.directory, sessionID: event.sessionID,
                messages: event.messages, summaryModelLimit: summary.modelContextLimit,
                summaryVariant: summary.summaryVariant, generate: summary.generate,
            }) }
        })
    },
})
