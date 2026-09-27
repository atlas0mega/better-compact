import { Rpc } from "@opencode/plugin/rpc"
import type { Context } from "@opencode/plugin/promise/plugin"
import { expireV2Archives, loadV2Catalog } from "./archive"

export const BetterCompactRPC = Rpc.define({
    id: "better-compact",
    methods: {
        status: {
            input: { type: "object", properties: { sessionID: { type: "string" } },
                required: ["sessionID"], additionalProperties: false },
            output: { type: "object", properties: {
                archiveCount: { type: "integer" }, readyCount: { type: "integer" },
                hasCheckpoint: { type: "boolean" }, retirementThrough: { type: "integer" },
                lastStatus: { type: "string" }, beforeTokens: { type: "integer" },
                afterTokens: { type: "integer" }, candidateTokens: { type: "integer" },
                validationFailure: { type: "string" }, triggerTokens: { type: "integer" },
                selectedStrategy: { type: "string" },
                handoffVisible: { type: "boolean" },
                targetTokens: { type: "integer" },
                reason: { type: "string" }, systemMessageCount: { type: "integer" },
                providerTokens: { type: "integer" }, contextLimit: { type: "integer" },
                summaryAvailable: { type: "boolean" },
                summaryFailure: { type: "string" },
                errorPhase: { type: "string" },
                errorSite: { type: "string" },
            }, required: ["archiveCount", "readyCount", "hasCheckpoint", "retirementThrough",
                "lastStatus", "beforeTokens", "afterTokens", "candidateTokens", "validationFailure", "selectedStrategy", "handoffVisible", "triggerTokens", "targetTokens",
                "reason", "systemMessageCount", "providerTokens", "contextLimit", "summaryAvailable", "summaryFailure", "errorPhase", "errorSite"],
                additionalProperties: false },
        },
    },
    events: {},
})

export async function registerV2RPC(ctx: Context): Promise<void> {
    await ctx.rpc.register(BetterCompactRPC, {
        status: async (raw) => {
            const sessionID = (raw as { sessionID?: unknown }).sessionID
            if (typeof sessionID !== "string" || !/^ses_[a-zA-Z0-9_-]{1,155}$/.test(sessionID))
                throw new Error("Invalid session ID")
            const session = await ctx.session.get({ sessionID })
            if (session.location.directory !== ctx.location.directory) throw new Error("Session belongs to another location")
            await expireV2Archives(session.location.directory, sessionID)
            const catalog = await loadV2Catalog(session.location.directory, sessionID)
            const stored = await ctx.storage.get(`status/${sessionID}`)
            const last: Record<string, unknown> = stored && typeof stored === "object" && !Array.isArray(stored)
                ? stored as Record<string, unknown> : {}
            const tokens = (key: string) => typeof last[key] === "number" && Number.isSafeInteger(last[key]) && last[key] >= 0
                ? last[key] as number : 0
            return { archiveCount: catalog.entries.filter((entry) => entry.status !== "expired").length,
                readyCount: catalog.entries.filter((entry) => entry.status === "ready").length,
                hasCheckpoint: !!catalog.checkpoint,
                retirementThrough: catalog.retirementThrough ?? 0,
                lastStatus: typeof last.status === "string" ? last.status : "unobserved",
                beforeTokens: tokens("beforeTokens"), afterTokens: tokens("afterTokens"),
                candidateTokens: tokens("candidateTokens"),
                validationFailure: typeof last.validationFailure === "string" ? last.validationFailure : "none",
                selectedStrategy: typeof last.selectedStrategy === "string" ? last.selectedStrategy : "none",
                handoffVisible: last.handoffVisible === true,
                triggerTokens: tokens("triggerTokens"), targetTokens: tokens("targetTokens"),
                reason: typeof last.reason === "string" ? last.reason : "none",
                systemMessageCount: tokens("systemMessageCount"),
                providerTokens: tokens("providerTokens"), contextLimit: tokens("contextLimit"),
                summaryAvailable: last.summaryAvailable === true,
                summaryFailure: catalog.summaryAttempt?.reason ?? "none",
                errorPhase: typeof last.errorPhase === "string" ? last.errorPhase : "none",
                errorSite: typeof last.errorSite === "string" ? last.errorSite : "none" }
        },
    })
}
