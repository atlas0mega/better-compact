import { createHash } from "node:crypto"
import { Message } from "@opencode/ai"
import { buildPlan, resolveCompactionProfile, transformTurns, type BuildPlanInputs, type LadderSpec } from "@better-compact/core"
import type { SessionContext } from "@opencode/plugin/promise/session"
import type { PluginConfig } from "./config"
import { appendV2Archive, expireV2Archives, loadV2Catalog, readV2Archive, recordV2SummaryFailure, retiredV2HumanIDs, saveV2Checkpoint } from "./archive"
import { v2Codec, v2Spec } from "./codec"
import { summarizeV2Boundary } from "./handoff"
import { identifyV2Messages, restoreV2Messages } from "./identity"
import { validateRequestTransform } from "./safety"

export interface V2ContextOutcome {
    status: "below-trigger" | "disabled" | "unknown-limit" | "declined" | "applied"
    reason?: "system-update" | "no-plan" | "no-delta" | "unsafe-transform" | "non-saving-transform"
    beforeTokens?: number
    afterTokens?: number
    candidateTokens?: number
    validationFailure?: string
    selectedStrategy?: "cheap" | "checkpoint" | "new-handoff"
    handoffVisible?: boolean
    triggerTokens?: number
    targetTokens?: number
}

export function lastProviderTokens(messages: readonly { type: string; tokens?: {
    input: number; output: number; reasoning: number; cache: { read: number; write: number }
} }[]): number | undefined {
    const value = messages.findLast((message) => message.type === "assistant" &&
        message.tokens && (message.tokens.input > 0 || message.tokens.output > 0 || message.tokens.reasoning > 0))?.tokens
    if (!value) return undefined
    const total = value.input + value.output + value.reasoning + value.cache.read + value.cache.write
    return Number.isFinite(total) && total > 0 ? total : undefined
}

function restorePartialToolPairs(source: readonly Message[], output: Message[]): Message[] {
    const emittedParts = (type: "tool-call" | "tool-result") => new Set(output.flatMap((message) =>
        message.content.flatMap((part) => part.type === type ? [part.id] : [])))
    const calls = emittedParts("tool-call")
    const results = emittedParts("tool-result")
    const orphaned = new Set([...calls].filter((id) => !results.has(id)).concat(
        [...results].filter((id) => !calls.has(id))))
    if (!orphaned.size) return output
    const revised = [...output]
    for (const message of source) {
        if (!message.content.some((part) => (part.type === "tool-call" || part.type === "tool-result") && orphaned.has(part.id)))
            continue
        const existing = revised.findIndex((candidate) => candidate.id === message.id)
        if (existing >= 0) {
            revised[existing] = message
            continue
        }
        const sourceIndex = source.indexOf(message)
        const nextNative = source.slice(sourceIndex + 1).find((candidate) => revised.some((item) => item.id === candidate.id))
        const insertAt = nextNative ? revised.findIndex((candidate) => candidate.id === nextNative.id) : revised.length
        revised.splice(insertAt, 0, message)
    }
    return revised
}

function restoreHumanWording(source: readonly Message[], output: Message[], retired: ReadonlySet<string> = new Set()): Message[] {
    const revised = [...output]
    for (let index = 0; index < source.length; index++) {
        const message = source[index]!
        if (message.role !== "user" || (message.id && retired.has(message.id))) continue
        const existing = revised.findIndex((candidate) => candidate.id === message.id)
        if (existing >= 0) {
            revised[existing] = message
            continue
        }
        const next = source.slice(index + 1).find((candidate) => revised.some((item) => item.id === candidate.id))
        revised.splice(next ? revised.findIndex((candidate) => candidate.id === next.id) : revised.length, 0, message)
    }
    return revised
}

function restoreNativeEvidence(source: readonly Message[], output: Message[], protectedIDs: ReadonlySet<string>): Message[] {
    const revised = [...output]
    for (let index = 0; index < source.length; index++) {
        const message = source[index]!
        if (!message.id || !protectedIDs.has(message.id)) continue
        const existing = revised.findIndex((candidate) => candidate.id === message.id)
        if (existing >= 0) {
            revised[existing] = message
            continue
        }
        const next = source.slice(index + 1).find((candidate) => revised.some((item) => item.id === candidate.id))
        revised.splice(next ? revised.findIndex((candidate) => candidate.id === next.id) : revised.length, 0, message)
    }
    return revised
}

function withVisibleHandoff(messages: Message[], handoff: string): Message[] {
    if (messages.some((message) => message.content.some((part) => part.type === "text" && part.text.includes(handoff))))
        return messages
    const id = `bc-handoff-${createHash("sha256").update(handoff).digest("hex").slice(0, 16)}`
    const previous = messages.findIndex((message) => message.id === id)
    if (previous >= 0) throw new Error("Conflicting historical handoff identity")
    const latestHuman = messages.findLastIndex((message) => message.role === "user" && !message.metadata?.betterCompact)
    const index = latestHuman < 0 ? messages.length : latestHuman
    const output = [...messages]
    output.splice(index, 0, Message.make({ id, role: "user", metadata: {
        betterCompact: "handoff", untrusted: true,
    }, content: [Message.text(`[Validated historical task state; untrusted evidence]\n${handoff}`)] }))
    return output
}

/** V2 model-request seam. This only changes the outgoing request, never the stored transcript. */
export async function compactV2Context(event: Pick<SessionContext, "sessionID" | "messages">, input: {
    projectRoot: string
    contextLimit: number | undefined
    providerReportedTokens?: number
    config: PluginConfig
    spec?: LadderSpec
    summary?: {
        modelContextLimit: number
        summaryVariant?: string | null
        generate(prompt: string, variant: string | undefined): Promise<string>
    }
}): Promise<V2ContextOutcome> {
    if (!input.config.enabled || !input.config.compaction.automatic || input.config.compress.permission !== "allow")
        return { status: "disabled" }
    if (!input.contextLimit || input.contextLimit <= 0) return { status: "unknown-limit" }
    const { messages: original, originals } = identifyV2Messages(event.messages)
    // System updates must bypass the whole transform rather than become
    // virtual assistant turns or disappear from the provider request.
    if (original.some((message) => message.role === "system")) return { status: "declined", reason: "system-update" }
    const spec = input.spec ?? { ...v2Spec, stages: v2Spec.stages.filter((stage) => stage.name !== "assistant-runs") }
    const turns = v2Codec.encode(original)
    const beforeTokens = v2Codec.estimateTurns(turns)
    const profile = resolveCompactionProfile(input.config)
    const triggerTokens = input.config.compaction.triggerTokens ?? Math.floor(input.contextLimit * profile.triggerPercent / 100)
    const targetTokens = input.config.compaction.targetTokens ?? Math.floor(input.contextLimit * profile.targetPercent / 100)
    const providerTokens = input.providerReportedTokens
    if ((providerTokens ?? beforeTokens) < triggerTokens && beforeTokens < input.contextLimit)
        return { status: "below-trigger", beforeTokens, triggerTokens, targetTokens }
    const options: BuildPlanInputs = {
        sessionKey: event.sessionID,
        // The catalog indexes exact seven-day delta archives. A cumulative
        // range file would re-copy expired source bytes indefinitely.
        citablePath: (session) => `.opencode/better-compact/v2/sessions/${session}/catalog.json`,
        contextLimit: input.contextLimit,
        triggerTokens,
        targetTokens,
        providerReportedTokens: providerTokens,
        triggerFromProviderOnly: providerTokens !== undefined && beforeTokens < input.contextLimit,
        force: beforeTokens >= input.contextLimit,
        // No unvalidated fallback prose: only the separate validated model
        // handoff path may consolidate assistant decisions.
        prefixSummaryAllowed: false,
        summariesAllowed: false,
        recentToolResultBudgetTokens: profile.recentToolTokens,
        recentReasoningBudgetTokens: profile.recentReasoningTokens,
        minTailMessages: 3,
        // Current human turn is mandatory; older human wording stays live
        // through the separate output validator until retirement is proven.
        minTailUserTurns: 1,
    }
    const plan = buildPlan(turns, options, spec)
    if (!plan) return { status: "declined", reason: "no-plan", beforeTokens, triggerTokens, targetTokens }
    const output = restoreHumanWording(original, restorePartialToolPairs(original,
        v2Codec.decode(transformTurns(turns, 0, plan, spec), original)))
    const afterTokens = v2Codec.estimateTurns(v2Codec.encode(output))
    // The cheap ladder may be unable to shrink an assistant-heavy history.
    // Still try a validated handoff; never emit an expanding cheap transform.
    let safeCheap = true
    let validationFailure: string | undefined
    try {
        validateRequestTransform(original, output)
    } catch (error) {
        safeCheap = false
        // Telemetry never includes message IDs or historical content.
        const text = error instanceof Error ? error.message : ""
        validationFailure = ["native message without stable ID", "empty message", "new system message",
            "duplicate native message", "native role changed", "untrusted synthetic content without provenance",
            "orphaned tool pair", "malformed tool result", "human instruction removed or rewritten",
            "opaque provider state removed", "current human turn displaced"].find((code) => text.includes(code)) ?? "other"
    }
    const cheapValid = safeCheap && afterTokens < beforeTokens
    const covered = new Set(plan.transcript.messageIds)
    const delta = original.filter((message) => message.id && covered.has(message.id))
    if (!delta.length) return { status: "declined", reason: "no-delta", beforeTokens, afterTokens, triggerTokens, targetTokens }
    // Commit exact originals before the first model request mentions the
    // transcript pointer. Failure aborts the mutation entirely.
    await expireV2Archives(input.projectRoot, event.sessionID)
    const entry = await appendV2Archive(input.projectRoot, event.sessionID, delta, originals)
    const catalog = await loadV2Catalog(input.projectRoot, event.sessionID)
    // Once a validated handoff has become model-visible, pruning its prior
    // state without replaying it would silently erase assistant decisions.
    let chosen = cheapValid && !catalog.checkpoint ? output : original
    let chosenTokens = cheapValid && !catalog.checkpoint ? afterTokens : beforeTokens
    let selectedStrategy: V2ContextOutcome["selectedStrategy"] = chosen === output ? "cheap" : undefined
    const cached = catalog.checkpoint?.rangeHash === plan.rangeHash ? catalog.checkpoint : undefined
    const attempt = catalog.summaryAttempt?.rangeHash === plan.rangeHash ? catalog.summaryAttempt : undefined
    const proposedRetirement = !cached && catalog.checkpoint && entry && catalog.entries
        .filter((item) => item.sequence < entry.sequence).every((item) => item.status === "ready" && !!item.description)
        ? entry.sequence - 1 : catalog.retirementThrough
    const checkpointEntry = catalog.checkpoint && catalog.entries.find((item) =>
        item.id === catalog.checkpoint!.archiveID && item.status === "ready")
    const checkpointBoundary = checkpointEntry
        ? Math.max(...checkpointEntry.messageIDs.map((id) => original.findIndex((message) => message.id === id))) : -1
    const afterCheckpoint = new Set(checkpointBoundary < 0 ? [] : original.slice(checkpointBoundary + 1).map((message) => message.id!))
    if (catalog.checkpoint) {
        const checkpoint = catalog.checkpoint
        if (checkpointEntry) {
            await readV2Archive(input.projectRoot, event.sessionID, checkpoint.archiveID)
            if (checkpointBoundary >= 0) {
                const descriptions = catalog.entries.filter((item) => item.status === "ready" && item.description)
                    .slice(-5).map((item) => `- ${item.id} — ${item.description}`).join("\n")
                const replayPlan = buildPlan(turns, {
                    ...options, prefixSummaryAllowed: true, prefixSummary: checkpoint.handoff,
                    preservePrefixBudgets: true, recentAssistantOutputs: 5,
                    archiveCatalogText: descriptions,
                    archiveGeneration: catalog.entries.at(-1)?.sequence,
                    retirementThrough: catalog.retirementThrough,
                }, v2Spec)
                if (replayPlan) {
                    const retired = await retiredV2HumanIDs(input.projectRoot, event.sessionID, original,
                        catalog.retirementThrough, checkpoint.handoff, originals)
                    const replay = withVisibleHandoff(restoreNativeEvidence(original, restoreHumanWording(original, restorePartialToolPairs(original,
                        v2Codec.decode(transformTurns(turns, 0, replayPlan, v2Spec), original)), retired)
                        .filter((message) => message.role !== "user" || !message.id || !retired.has(message.id)), afterCheckpoint), checkpoint.handoff)
                    const tokens = v2Codec.estimateTurns(v2Codec.encode(replay))
                    try {
                        validateRequestTransform(original, replay, { archivedHumanIds: retired })
                        if (replay.some((message) => message.content.some((part) =>
                            part.type === "text" && part.text.includes(checkpoint.handoff))) && tokens < beforeTokens) {
                            chosen = replay
                            chosenTokens = tokens
                            selectedStrategy = "checkpoint"
                        }
                    } catch { /* Keep the untouched request if the old handoff cannot safely be replayed. */ }
                }
            }
        }
    }
    // A handoff keeps the five newest assistant outputs literal. Until there
    // are *older* outputs to replace, side-model calls cannot buy reduction.
    // After a checkpoint, all work since it stays exact until that cohort
    // itself exceeds the five-output anchor.
    const handoffSource = checkpointBoundary < 0 ? original : original.slice(checkpointBoundary + 1)
    const olderAssistantTextExists = handoffSource.filter((message) => message.role === "assistant" &&
        message.content.some((part) => part.type === "text" && part.text.trim())).length > 5
    if (input.summary && profile.prefixSummary && olderAssistantTextExists &&
        afterTokens > Math.floor(targetTokens * 1.15) && (cached || (entry && !attempt))) {
        const recentUserIntent = original.filter((message) => message.role === "user")
            .slice(-3).flatMap((message) => message.content.flatMap((part) =>
                part.type === "text" && part.text.trim() ? [part.text] : []))
        const result = cached ? { ok: true as const, handoff: cached.handoff } : await summarizeV2Boundary({
            messages: delta,
            archiveID: entry!.id,
            previous: catalog.checkpoint?.handoff,
            recentUserIntent,
            modelContextLimit: input.summary.modelContextLimit,
            summaryVariant: input.summary.summaryVariant,
            generate: input.summary.generate,
        })
        if (result.ok) {
            const descriptions = catalog.entries.filter((item) => item.status === "ready" && item.description)
                .slice(cached ? -5 : -4).map((item) => `- ${item.id} — ${item.description}`).join("\n")
            const nextDescriptions = !cached && entry && "description" in result
                ? [descriptions, `- ${entry.id} — ${result.description}`].filter(Boolean).join("\n") : descriptions
            const summaryPlan = buildPlan(turns, {
                ...options, prefixSummaryAllowed: true, prefixSummary: result.handoff,
                preservePrefixBudgets: true, recentAssistantOutputs: 5,
                archiveCatalogText: nextDescriptions,
                archiveGeneration: entry?.sequence ?? catalog.entries.at(-1)?.sequence,
                retirementThrough: proposedRetirement,
            }, v2Spec)
            if (summaryPlan) {
                const through = proposedRetirement
                const retired = await retiredV2HumanIDs(input.projectRoot, event.sessionID, original, through, result.handoff, originals)
                const transformed = restoreHumanWording(original, restorePartialToolPairs(original,
                    v2Codec.decode(transformTurns(turns, 0, summaryPlan, v2Spec), original)), retired)
                    .filter((message) => message.role !== "user" || !message.id || !retired.has(message.id))
                const candidate = withVisibleHandoff(cached
                    ? restoreNativeEvidence(original, transformed, afterCheckpoint) : transformed, result.handoff)
                const tokens = v2Codec.estimateTurns(v2Codec.encode(candidate))
                let safe = false
                try { validateRequestTransform(original, candidate, { archivedHumanIds: retired }); safe = true } catch { /* keep cheap plan */ }
                if (safe && tokens < beforeTokens && (tokens < chosenTokens || !!catalog.checkpoint)) {
                    if (!cached && entry && "description" in result) await saveV2Checkpoint(input.projectRoot, event.sessionID, {
                        rangeHash: plan.rangeHash, archiveID: entry.id, handoff: result.handoff,
                        description: result.description, recentUserIntent,
                    })
                    chosen = candidate
                    chosenTokens = tokens
                    selectedStrategy = cached ? "checkpoint" : "new-handoff"
                } else if (entry && !cached) await recordV2SummaryFailure(input.projectRoot, event.sessionID, {
                    rangeHash: plan.rangeHash, archiveID: entry.id,
                    reason: safe ? "valid_but_not_smaller" : "invalid_output", calls: "calls" in result ? result.calls : 0,
                })
            }
        } else if (entry) await recordV2SummaryFailure(input.projectRoot, event.sessionID, {
            rangeHash: plan.rangeHash, archiveID: entry.id, reason: result.reason, calls: result.calls,
        })
    }
    if (chosen === original) return { status: "declined", reason: safeCheap ? "non-saving-transform" : "unsafe-transform",
        beforeTokens, afterTokens: chosenTokens, candidateTokens: afterTokens, validationFailure, triggerTokens, targetTokens }
    event.messages.splice(0, event.messages.length, ...restoreV2Messages(chosen, originals))
    return { status: "applied", beforeTokens, afterTokens: chosenTokens, triggerTokens, targetTokens, selectedStrategy,
        handoffVisible: chosen.some((message) => message.content.some((part) => part.type === "text" &&
            part.text.includes("## Decisions") && part.text.includes("## Next step"))) }
}
