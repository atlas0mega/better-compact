import { Plugin } from "@opencode/plugin/tui"
import { BetterCompactRPC } from "./rpc"

export default Plugin.define({
    id: "better-compact.tui",
    setup(ctx) {
        const rpc = ctx.client.rpc(BetterCompactRPC)
        ctx.keymap.layer(() => ({
            mode: "global",
            commands: [{
                id: "better-compact.run",
                title: "Request Better Compact checkpoint",
                group: "Better Compact",
                palette: true,
                slash: { name: "better-compact" },
                run: async () => {
                    const route = ctx.ui.router.current()
                    if (route.type !== "session") {
                        ctx.ui.toast.show({ title: "Better Compact", message: "Open a session first.", variant: "warning" })
                        return
                    }
                    try {
                        await rpc.status({ sessionID: route.sessionID })
                        await ctx.client.session.compact({ sessionID: route.sessionID })
                        ctx.ui.toast.show({ title: "Better Compact",
                            message: "Compaction queued at the next safe point; originals remain archived.", variant: "info" })
                    } catch {
                        ctx.ui.toast.show({ title: "Better Compact",
                            message: "Compaction request failed; original context was not replaced.", variant: "warning" })
                    }
                },
            }, {
                id: "better-compact.status",
                title: "Better Compact archive status",
                group: "Better Compact",
                palette: true,
                slash: { name: "better-compact-status" },
                run: async () => {
                    const route = ctx.ui.router.current()
                    if (route.type !== "session") {
                        ctx.ui.toast.show({ title: "Better Compact", message: "Open a session first.", variant: "warning" })
                        return
                    }
                    try {
                        const state = await rpc.status({ sessionID: route.sessionID })
                        if (!state || typeof state !== "object" ||
                            !("archiveCount" in state) || typeof state.archiveCount !== "number" ||
                            !("readyCount" in state) || typeof state.readyCount !== "number" ||
                            !("hasCheckpoint" in state) || typeof state.hasCheckpoint !== "boolean" ||
                            !("lastStatus" in state) || typeof state.lastStatus !== "string")
                            throw new Error("Invalid Better Compact status")
                        ctx.ui.toast.show({ title: "Better Compact",
                            message: `${state.archiveCount} exact archives (${state.readyCount} described); ${state.hasCheckpoint ? "validated handoff" : "no handoff yet"}. Last request: ${state.lastStatus}${"reason" in state && typeof state.reason === "string" && state.reason !== "none" ? ` (${state.reason})` : ""}${"providerTokens" in state && "triggerTokens" in state ? ` (${state.providerTokens}/${state.triggerTokens} provider/trigger tokens)` : ""}${"summaryAvailable" in state && state.summaryAvailable === false && state.lastStatus !== "error" ? "; configured summary model unavailable" : ""}${"summaryFailure" in state && typeof state.summaryFailure === "string" && state.summaryFailure !== "none" ? `; last summary: ${state.summaryFailure}` : ""}.`,
                            variant: "info" })
                    } catch {
                        ctx.ui.toast.show({ title: "Better Compact",
                            message: "Status unavailable from the active server plugin.", variant: "warning" })
                    }
                },
            }],
        }))
    },
})
