import { createHash } from "node:crypto"
import { Message } from "@opencode/ai"
import { buildPlan, countTokens, resolveCompactionProfile, transformTurns, type BuildPlanInputs, type LadderSpec } from "@better-compact/core"
import type { SessionContext } from "@opencode/plugin/promise/session"
import type { PluginConfig } from "./config"
import { appendV2Archive, expireV2Archives, loadV2Catalog, readV2Archive, recordV2SummaryFailure, retiredV2HumanIDs, saveV2Checkpoint, saveV2ReplayFrontier, v2ReplayPrefixHash } from "./archive"
import { v2Codec, v2Spec } from "./codec"
import { summarizeV2Boundary } from "./handoff"
import { identifyV2Messages, restoreV2Messages } from "./identity"
import { validateRequestTransform } from "./safety"

export interface V2ContextOutcome {
    status: "below-trigger" | "disabled" | "unknown-limit" | "declined" | "applied"
    reason?: "system-update" | "no-plan" | "no-delta" | "replay-frontier-missing" | "unsafe-transform" |
        "non-saving-transform" | "needs-native-fallback"
    beforeTokens?: number
    afterTokens?: number
    candidateTokens?: number
    validationFailure?: string
    selectedStrategy?: "cheap" | "checkpoint" | "new-handoff"
    handoffVisible?: boolean
    triggerTokens?: number
    targetTokens?: number
    /** Whole canonical provider messages, starting at the earliest protected
     * user/system/replay boundary, for an optional selective native fallback. */
    protectedTailMessages?: number
}

export type V2ContextPhase = "identity" | "estimate" | "plan-build" | "plan-transform" |
    "plan-decode" | "plan-price" | "plan-validate" | "plan-delta" | "archive-expiry" | "archive-write" |
    "catalog" | "checkpoint" | "handoff" | "commit"

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
    /** Host preflight threshold. No outgoing request or checkpoint is committed
     * if even the validated candidate cannot fit beneath this ceiling. */
    fallbackCeiling?: number
    config: PluginConfig
    spec?: LadderSpec
    summary?: {
        modelContextLimit: number
        summaryVariant?: string | null
        generate(prompt: string, variant: string | undefined): Promise<string>
    }
    trace?: (phase: V2ContextPhase) => void
}): Promise<V2ContextOutcome> {
    if (!input.config.enabled || !input.config.compaction.automatic || input.config.compress.permission !== "allow")
        return { status: "disabled" }
    if (!input.contextLimit || input.contextLimit <= 0) return { status: "unknown-limit" }
    input.trace?.("identity")
    const { messages: original, originals } = identifyV2Messages(event.messages)
    // A chronological system update changes instruction authority. Keep the
    // whole prefix through the last system update byte-exact and in place;
    // only the later conversation can be planned, archived or summarized.
    const barrier = original.findLastIndex((message) => message.role === "system")
    const immutableHead = original.slice(0, barrier + 1)
    const active = original.slice(barrier + 1)
    const authorityKey = barrier < 0 ? undefined : createHash("sha256").update(JSON.stringify(
        immutableHead.map((message) => originals.get(message.id!) ?? message))).digest("hex")
    const hasActiveHuman = active.some((message) => message.role === "user")
    // Tool loops can continue after a system update without another human
    // message. Cheap tool pruning is still safe because the prior human input
    // stays exact in the authority prefix; a model handoff, however, must wait
    // for a new user turn in this epoch.
    if (!hasActiveHuman && (!original.some((message) => message.role === "user") ||
        !active.some((message) => message.role === "tool")))
        return { status: "declined", reason: "system-update" }
    const spec = input.spec ?? { ...v2Spec, stages: v2Spec.stages.filter((stage) => stage.name !== "assistant-runs") }
    input.trace?.("estimate")
    const turns = v2Codec.encode(active)
    const headTokens = immutableHead.length ? countTokens(JSON.stringify(immutableHead.map((message) =>
        ({ role: message.role, content: message.content })))) : 0
    const estimate = (messages: readonly Message[]) => headTokens +
        v2Codec.estimateTurns(v2Codec.encode(messages.slice(immutableHead.length)))
    const beforeTokens = headTokens + v2Codec.estimateTurns(turns)
    const profile = resolveCompactionProfile(input.config)
    const triggerTokens = input.config.compaction.triggerTokens ?? Math.floor(input.contextLimit * profile.triggerPercent / 100)
    const targetTokens = input.config.compaction.targetTokens ?? Math.floor(input.contextLimit * profile.targetPercent / 100)
    const providerTokens = input.providerReportedTokens
    if ((providerTokens ?? beforeTokens) < triggerTokens && beforeTokens < input.contextLimit &&
        (input.fallbackCeiling === undefined || beforeTokens < input.fallbackCeiling))
        return { status: "below-trigger", beforeTokens, triggerTokens, targetTokens }
    if (headTokens >= input.contextLimit) return { status: "declined", reason: "system-update",
        beforeTokens, triggerTokens, targetTokens }
    // Request transforms are not durable transcript rewrites. Once an exact
    // prefix was archived, the next request must replay *all* work since that
    // first boundary natively, including assistant/tool continuations without
    // another human turn. Never let target chasing advance into that cohort.
    const previousCatalog = await loadV2Catalog(input.projectRoot, event.sessionID)
    const priorCheckpointSameAuthority = previousCatalog.checkpoint?.authorityKey === authorityKey &&
        !!previousCatalog.checkpoint
    // Preparing a private archive is not proof a transform ever reached the
    // provider. In particular a young assistant-heavy session may have many
    // pending archives before a handoff is finally validated. Old catalogs
    // without replayState are handled conservatively as potentially applied.
    const activeIDs = new Set(active.map((message) => message.id!))
    const firstArchive = previousCatalog.replayState === "unapplied" ? undefined :
        previousCatalog.entries.find((entry) => entry.status !== "expired" && entry.authorityKey === authorityKey &&
            entry.messageIDs.some((id) => activeIDs.has(id)))
    const archivedIDs = new Set(firstArchive?.messageIDs ?? [])
    const oldAnchor = previousCatalog.replayFrontier?.anchors.at(-1)
    const sameAuthority = previousCatalog.replayFrontier?.authorityKey === authorityKey
    const replayAnchor = sameAuthority && oldAnchor && activeIDs.has(oldAnchor.id) ? oldAnchor : undefined
    const hashOf = (message: Message) => createHash("sha256").update(JSON.stringify(originals.get(message.id!) ?? message)).digest("hex")
    // A matching last ID is not proof: earlier native text can change while
    // preserving that ID. Never replay an older handoff over revised evidence.
    const verified = replayAnchor ? active.findIndex((message) => message.id === replayAnchor.id &&
        hashOf(message) === replayAnchor.sha256) : -1
    // An older replay frontier before a new system barrier belongs to the
    // previous authority epoch. Do not reuse its checkpoint in this epoch.
    if (previousCatalog.replayFrontier && replayAnchor && (verified < 0 ||
        v2ReplayPrefixHash([...immutableHead, ...active.slice(0, verified + 1)], originals) !== previousCatalog.replayFrontier.prefixSha256))
        return { status: "declined", reason: "replay-frontier-missing", beforeTokens, triggerTokens, targetTokens }
    if (previousCatalog.replayFrontier && sameAuthority && !replayAnchor)
        return { status: "declined", reason: "replay-frontier-missing", beforeTokens, triggerTokens, targetTokens }
    if (replayAnchor && !firstArchive)
        return { status: "declined", reason: "replay-frontier-missing", beforeTokens, triggerTokens, targetTokens }
    const anchor = verified >= 0 ? verified : firstArchive
        ? active.findLastIndex((message) => !!message.id && archivedIDs.has(message.id)) : -1
    if (firstArchive && anchor < 0)
        return { status: "declined", reason: "replay-frontier-missing", beforeTokens, triggerTokens, targetTokens }
    const frontier = firstArchive ? anchor + 1 : turns.length
    const planningTurns = turns.slice(0, frontier)
    const rawSinceReplay = active.slice(frontier)
    const latestHuman = original.findLastIndex((message) => message.role === "user")
    const authorityOrHuman = latestHuman > barrier ? latestHuman : Math.max(0, barrier)
    // Native checkpoints are replay boundaries too. After a checkpoint, the
    // *whole* subsequently appended cohort (not merely the newest human
    // prompt) is protected, including newer encrypted assistant reasoning.
    const nativeCheckpoint = original.findLastIndex((message) => message.content.some((part) => part.type === "compaction"))
    const protectedTailMessages = original.length - Math.min(authorityOrHuman, immutableHead.length + frontier,
        nativeCheckpoint < 0 ? original.length : nativeCheckpoint + 1)
    const options: BuildPlanInputs = {
        sessionKey: event.sessionID,
        // The catalog indexes exact seven-day delta archives. A cumulative
        // range file would re-copy expired source bytes indefinitely.
        citablePath: (session) => `.opencode/better-compact/v2/sessions/${session}/catalog.json`,
        contextLimit: input.contextLimit - headTokens,
        triggerTokens: Math.max(1, triggerTokens - headTokens),
        targetTokens: Math.max(1, targetTokens - headTokens),
        providerReportedTokens: providerTokens ? Math.max(1, providerTokens - headTokens) : undefined,
        triggerFromProviderOnly: providerTokens !== undefined && beforeTokens < input.contextLimit,
        // The full request has already crossed the trigger. A replay prefix
        // alone may now sit below the trigger (especially on a later, larger
        // model window); still reconstruct its archived transform.
        force: beforeTokens >= input.contextLimit || !!firstArchive || immutableHead.length > 0 ||
            input.fallbackCeiling !== undefined && beforeTokens >= input.fallbackCeiling,
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
        // The core's generic per-run reference index repeats every archived
        // assistant/tool span (hundreds of lines in real sessions). V2 has a
        // verified session-owned recall catalog instead; omit that duplicate
        // index until there are validated archive descriptions to display.
        archiveCatalogText: "",
    }
    input.trace?.("plan-build")
    const plan = buildPlan(planningTurns, options, spec)
    if (!plan) return { status: "declined", reason: "no-plan", beforeTokens, triggerTokens, targetTokens,
        protectedTailMessages }
    input.trace?.("plan-transform")
    const transformed = transformTurns(planningTurns, 0, plan, spec)
    input.trace?.("plan-decode")
    const output = restoreHumanWording(original, restorePartialToolPairs(original,
        [...immutableHead, ...v2Codec.decode(transformed, original), ...rawSinceReplay]))
    input.trace?.("plan-price")
    const afterTokens = estimate(output)
    // The cheap ladder may be unable to shrink an assistant-heavy history.
    // Still try a validated handoff; never emit an expanding cheap transform.
    let safeCheap = true
    let validationFailure: string | undefined
    input.trace?.("plan-validate")
    try {
        validateRequestTransform(original, output)
    } catch (error) {
        safeCheap = false
        // Telemetry never includes message IDs or historical content.
        const text = error instanceof Error ? error.message : ""
        validationFailure = ["native message without stable ID", "empty message", "new system message",
            "duplicate native message", "native role changed", "untrusted synthetic content without provenance",
            "orphaned tool pair", "malformed tool result", "human instruction removed or rewritten",
            "opaque provider state removed", "current human turn displaced", "system authority prefix changed"].find((code) => text.includes(code)) ?? "other"
    }
    const cheapValid = safeCheap && afterTokens < beforeTokens
    const possibleHandoff = hasActiveHuman && !!input.summary && profile.prefixSummary &&
        active.filter((message) => message.role === "assistant" && message.content.some((part) =>
            part.type === "text" && part.text.trim())).length > 5 && afterTokens > Math.floor(targetTokens * 1.15)
    // No archive should be published for a request which cannot be changed
    // (nor generate a validated handoff). Otherwise an unapplied pending
    // archive would incorrectly become the replay boundary next request.
    if (!cheapValid && !possibleHandoff && !priorCheckpointSameAuthority)
        return { status: "declined", reason: safeCheap ? "non-saving-transform" : "unsafe-transform",
            beforeTokens, afterTokens, validationFailure, triggerTokens, targetTokens, protectedTailMessages }
    input.trace?.("plan-delta")
    const covered = new Set(plan.transcript.messageIds)
    const delta = active.filter((message) => message.id && covered.has(message.id))
    if (!delta.length) return { status: "declined", reason: "no-delta", beforeTokens, afterTokens, triggerTokens, targetTokens,
        protectedTailMessages }
    // Commit exact originals before the first model request mentions the
    // transcript pointer. Failure aborts the mutation entirely.
    input.trace?.("archive-expiry")
    await expireV2Archives(input.projectRoot, event.sessionID)
    input.trace?.("archive-write")
    const entry = cheapValid || possibleHandoff && !priorCheckpointSameAuthority
        ? await appendV2Archive(input.projectRoot, event.sessionID, delta, originals, authorityKey) : null
    input.trace?.("catalog")
    const catalog = await loadV2Catalog(input.projectRoot, event.sessionID)
    const activeCheckpoint = catalog.checkpoint?.authorityKey === authorityKey && catalog.entries.some((item) =>
        item.id === catalog.checkpoint?.archiveID && item.messageIDs.some((id) => activeIDs.has(id)))
        ? catalog.checkpoint : undefined
    // Once a validated handoff has become model-visible, pruning its prior
    // state without replaying it would silently erase assistant decisions.
    let chosen = cheapValid && !activeCheckpoint ? output : original
    let chosenTokens = cheapValid && !activeCheckpoint ? afterTokens : beforeTokens
    let selectedStrategy: V2ContextOutcome["selectedStrategy"] = chosen === output ? "cheap" : undefined
    let pendingCheckpoint: { rangeHash: string; archiveID: string; handoff: string; description: string;
        recentUserIntent: string[]; authorityKey?: string } | undefined
    let pendingCheckpointCalls = 0
    const checkpointEntry = activeCheckpoint && catalog.entries.find((item) =>
        item.id === activeCheckpoint.archiveID && item.status === "ready")
    const checkpointBoundary = checkpointEntry
        ? Math.max(...checkpointEntry.messageIDs.map((id) => original.findIndex((message) => message.id === id))) : -1
    const afterCheckpoint = new Set(checkpointBoundary < 0 ? [] : original.slice(checkpointBoundary + 1).map((message) => message.id!))
    input.trace?.("checkpoint")
    if (activeCheckpoint) {
        const checkpoint = activeCheckpoint
        if (checkpointEntry) {
            await readV2Archive(input.projectRoot, event.sessionID, checkpoint.archiveID)
            if (checkpointBoundary >= 0) {
                const descriptions = catalog.entries.filter((item) => item.status === "ready" && item.description)
                    .slice(-5).map((item) => `- ${item.id} — ${item.description}`).join("\n")
                const replayPlan = buildPlan(planningTurns, {
                    ...options, prefixSummaryAllowed: true, prefixSummary: checkpoint.handoff,
                    preservePrefixBudgets: true, recentAssistantOutputs: 5,
                    archiveCatalogText: descriptions,
                    archiveGeneration: catalog.entries.at(-1)?.sequence,
                    retirementThrough: catalog.retirementThrough,
                }, v2Spec)
                if (replayPlan) {
                    const retired = await retiredV2HumanIDs(input.projectRoot, event.sessionID, original,
                        catalog.retirementThrough, checkpoint.handoff, originals, authorityKey)
                    const replay = withVisibleHandoff(restoreNativeEvidence(original, restoreHumanWording(original, restorePartialToolPairs(original,
                        [...immutableHead, ...v2Codec.decode(transformTurns(planningTurns, 0, replayPlan, v2Spec), original), ...rawSinceReplay]), retired)
                        .filter((message) => message.role !== "user" || !message.id || !retired.has(message.id)), afterCheckpoint), checkpoint.handoff)
                    const tokens = estimate(replay)
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
    const handoffSource = checkpointBoundary < 0 ? active : original.slice(checkpointBoundary + 1)
    const olderAssistantTextExists = handoffSource.filter((message) => message.role === "assistant" &&
        message.content.some((part) => part.type === "text" && part.text.trim())).length > 5
    input.trace?.("handoff")
    // The cheap plan never crosses the replay frontier. A *new*, validated
    // checkpoint may cover newly accumulated work, but it must archive that
    // work first and validate against the entire native request. The full
    // source fingerprint also prevents paying Luna twice for a failed replay.
    const newCohort = !!activeCheckpoint && rawSinceReplay.length > 0 && olderAssistantTextExists
    const summarySource = newCohort ? original.slice(checkpointBoundary + 1) : delta
    const summaryTurns = newCohort ? turns : planningTurns
    const summaryKey = newCohort ? createHash("sha256").update(JSON.stringify(original.map((message) =>
        originals.get(message.id!) ?? message))).digest("hex").slice(0, 16) : plan.rangeHash
    const summaryCached = activeCheckpoint?.rangeHash === summaryKey ? activeCheckpoint : undefined
    const summaryAttempt = catalog.summaryAttempt?.authorityKey === authorityKey &&
        catalog.summaryAttempt?.rangeHash === summaryKey ? catalog.summaryAttempt : undefined
    if (hasActiveHuman && input.summary && profile.prefixSummary && olderAssistantTextExists &&
        afterTokens > Math.floor(targetTokens * 1.15) && (summaryCached || !summaryAttempt)) {
        const summaryEntry = newCohort
            ? await appendV2Archive(input.projectRoot, event.sessionID, summarySource, originals, authorityKey)
            : entry
        if (newCohort && !summaryEntry) return { status: "declined", reason: "no-delta", beforeTokens, afterTokens,
            triggerTokens, targetTokens, protectedTailMessages }
        const retirement = !summaryCached && activeCheckpoint && summaryEntry && catalog.entries
            .some((item) => item.authorityKey === authorityKey && item.sequence < summaryEntry.sequence) && catalog.entries
            .filter((item) => item.authorityKey === authorityKey && item.sequence < summaryEntry.sequence)
            .every((item) => item.status === "ready" && !!item.description)
            ? summaryEntry.sequence - 1 : activeCheckpoint ? catalog.retirementThrough : undefined
        const recentUserIntent = active.filter((message) => message.role === "user")
            .slice(-3).flatMap((message) => message.content.flatMap((part) =>
                part.type === "text" && part.text.trim() ? [part.text] : []))
        const result = summaryCached ? { ok: true as const, handoff: summaryCached.handoff } : await summarizeV2Boundary({
            messages: summarySource,
            archiveID: summaryEntry!.id,
            previous: activeCheckpoint?.handoff,
            recentUserIntent,
            modelContextLimit: input.summary.modelContextLimit,
            summaryVariant: input.summary.summaryVariant,
            generate: input.summary.generate,
        })
        if (result.ok) {
            const descriptions = catalog.entries.filter((item) => item.status === "ready" && item.description)
                .slice(summaryCached ? -5 : -4).map((item) => `- ${item.id} — ${item.description}`).join("\n")
            const nextDescriptions = !summaryCached && summaryEntry && "description" in result
                ? [descriptions, `- ${summaryEntry.id} — ${result.description}`].filter(Boolean).join("\n") : descriptions
            const summaryPlan = buildPlan(summaryTurns, {
                ...options, prefixSummaryAllowed: true, prefixSummary: result.handoff,
                preservePrefixBudgets: true, recentAssistantOutputs: 5,
                archiveCatalogText: nextDescriptions,
                archiveGeneration: summaryEntry?.sequence ?? catalog.entries.at(-1)?.sequence,
                retirementThrough: retirement,
            }, v2Spec)
            if (summaryPlan) {
                const retired = await retiredV2HumanIDs(input.projectRoot, event.sessionID, original,
                    retirement, result.handoff, originals, authorityKey)
                const transformed = restoreHumanWording(original, restorePartialToolPairs(original,
                    [...immutableHead, ...v2Codec.decode(transformTurns(summaryTurns, 0, summaryPlan, v2Spec), original),
                        ...(newCohort ? [] : rawSinceReplay)]), retired)
                    .filter((message) => message.role !== "user" || !message.id || !retired.has(message.id))
                const candidate = withVisibleHandoff(summaryCached
                    ? restoreNativeEvidence(original, transformed, afterCheckpoint) : transformed, result.handoff)
                const tokens = estimate(candidate)
                let safe = false
                try { validateRequestTransform(original, candidate, { archivedHumanIds: retired }); safe = true } catch { /* keep cheap plan */ }
                if (safe && tokens < beforeTokens && (tokens < chosenTokens || !!activeCheckpoint)) {
                    if (!summaryCached && summaryEntry && "description" in result) pendingCheckpoint = {
                        rangeHash: summaryKey, archiveID: summaryEntry.id, handoff: result.handoff,
                        description: result.description, recentUserIntent, authorityKey,
                    }
                    if (pendingCheckpoint && "calls" in result) pendingCheckpointCalls = result.calls
                    chosen = candidate
                    chosenTokens = tokens
                    selectedStrategy = summaryCached ? "checkpoint" : "new-handoff"
                } else if (summaryEntry && !summaryCached) await recordV2SummaryFailure(input.projectRoot, event.sessionID, {
                    rangeHash: summaryKey, archiveID: summaryEntry.id,
                    reason: safe ? "valid_but_not_smaller" : "invalid_output", calls: "calls" in result ? result.calls : 0,
                    authorityKey,
                })
            }
        } else if (summaryEntry) await recordV2SummaryFailure(input.projectRoot, event.sessionID, {
            rangeHash: summaryKey, archiveID: summaryEntry.id, reason: result.reason, calls: result.calls, authorityKey,
        })
    }
    if (input.fallbackCeiling !== undefined && chosenTokens >= input.fallbackCeiling) {
        if (pendingCheckpoint) await recordV2SummaryFailure(input.projectRoot, event.sessionID, {
            rangeHash: pendingCheckpoint.rangeHash, archiveID: pendingCheckpoint.archiveID,
            reason: "needs_native_fallback", calls: pendingCheckpointCalls, authorityKey,
        })
        return { status: "declined", reason: "needs-native-fallback", beforeTokens,
            afterTokens: chosenTokens, candidateTokens: afterTokens, triggerTokens, targetTokens, protectedTailMessages }
    }
    if (chosen === original) return { status: "declined", reason: safeCheap ? "non-saving-transform" : "unsafe-transform",
        beforeTokens, afterTokens: chosenTokens, candidateTokens: afterTokens, validationFailure, triggerTokens, targetTokens,
        protectedTailMessages }
    if (pendingCheckpoint) await saveV2Checkpoint(input.projectRoot, event.sessionID, pendingCheckpoint)
    const newAuthorityEpoch = !!previousCatalog.replayFrontier && !sameAuthority
    if (!previousCatalog.replayFrontier || selectedStrategy === "new-handoff" || newAuthorityEpoch) {
        input.trace?.("catalog")
        await saveV2ReplayFrontier(input.projectRoot, event.sessionID,
            firstArchive && (!previousCatalog.replayFrontier || newAuthorityEpoch)
                ? [...immutableHead, ...active.slice(0, frontier)] : original,
            originals, selectedStrategy === "new-handoff" || newAuthorityEpoch, authorityKey)
    }
    input.trace?.("commit")
    event.messages.splice(0, event.messages.length, ...restoreV2Messages(chosen, originals))
    return { status: "applied", beforeTokens, afterTokens: chosenTokens, triggerTokens, targetTokens, selectedStrategy,
        protectedTailMessages,
        handoffVisible: chosen.some((message) => message.content.some((part) => part.type === "text" &&
            part.text.includes("## Decisions") && part.text.includes("## Next step"))) }
}
