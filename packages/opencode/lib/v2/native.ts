import type { Message } from "@opencode/ai"
import { countTokens, rangeHash } from "@better-compact/core"
import { appendV2Archive, loadV2Catalog, saveV2Checkpoint } from "./archive"
import { v2Codec } from "./codec"
import { summarizeV2Boundary } from "./handoff"
import { identifyV2Messages } from "./identity"

/** Opaque checkpoints need the host/provider compaction path, not a text handoff. */
export function hasOpaqueProviderState(messages: readonly Message[]): boolean {
    return messages.some((message) => message.content.some((part) => part.type === "compaction" ||
        part.type === "reasoning" && !!part.encrypted))
}

/** Native V2 manual-compaction hook: accept only a complete, exact-user-preserving handoff. */
export async function compactV2Native(input: {
    root: string
    sessionID: string
    messages: readonly Message[]
    summaryModelLimit: number
    summaryVariant?: string | null
    generate(prompt: string, variant: string | undefined): Promise<string>
}): Promise<string> {
    const { messages: original, originals } = identifyV2Messages(input.messages)
    if (original.some((message) => message.role === "system"))
        throw new Error("Cannot safely compact V2 system updates")
    if (hasOpaqueProviderState(original))
        throw new Error("Opaque provider state requires native provider compaction")
    const originalTokens = v2Codec.estimateTurns(v2Codec.encode(original))
    const catalogBefore = await loadV2Catalog(input.root, input.sessionID)
    const entry = await appendV2Archive(input.root, input.sessionID, original, originals)
    if (!entry) throw new Error("No new exact V2 material for manual compaction")
    const intent = original.filter((message) => message.role === "user").flatMap((message) =>
        message.content.flatMap((part) => part.type === "text" && part.text.trim() ? [part.text] : []))
    const summary = await summarizeV2Boundary({
        messages: original,
        archiveID: entry.id,
        previous: catalogBefore.checkpoint?.handoff,
        recentUserIntent: intent,
        modelContextLimit: input.summaryModelLimit,
        summaryVariant: input.summaryVariant,
        generate: input.generate,
    })
    if (!summary.ok) throw new Error(`V2 handoff declined: ${summary.reason}`)
    if (countTokens(summary.handoff) >= originalTokens) throw new Error("V2 handoff would not reduce the provider request")
    await saveV2Checkpoint(input.root, input.sessionID, {
        rangeHash: rangeHash(v2Codec.encode(original)), archiveID: entry.id,
        handoff: summary.handoff, description: summary.description, recentUserIntent: intent,
    })
    return summary.handoff
}
