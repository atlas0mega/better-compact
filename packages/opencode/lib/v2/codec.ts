import { createHash } from "node:crypto"
import { Message, type ContentPart } from "@opencode/ai"
import {
    assistantRunsStage, countTokens, purgeErrorInputsStage, reasoningStage, skillsStage,
    supersedeReadsStage, toolsOldStage, toolsRemainingStage,
    type Codec, type Conventions, type Item, type LadderSpec, type Turn,
} from "@better-compact/core"

const key = (value: unknown) => createHash("sha256").update(JSON.stringify(value)).digest("hex").slice(0, 16)
const partOf = (item: Exclude<Item, { kind: "synthetic" }>) => item.handle as ContentPart

function encodePart(part: ContentPart, messageKey: string, index: number): Item {
    const itemKey = `${messageKey}:${index}`
    if (part.type === "text") return { kind: "text", key: itemKey, text: part.text, handle: part }
    if (part.type === "reasoning") return { kind: "reasoning", key: itemKey, handle: part }
    if (part.type === "tool-call" || part.type === "tool-result") {
        return { kind: "tool", key: itemKey, callId: part.id, handle: part }
    }
    return { kind: "opaque", key: itemKey, handle: part }
}

// V2's model-request messages are not the persisted session-context records.
// Handles are immutable V2 content, and encrypted provider reasoning/checkpoints remain opaque.
export const v2Codec: Codec<Message> = {
    encode(messages) {
        // Chronological system updates carry operator authority and cannot be
        // converted into assistant turns. Until the V2 ladder has a protected
        // system-turn contract, refuse this request instead of dropping one.
        if (messages.some((message) => message.role === "system")) {
            throw new Error("V2 system-update messages must bypass Better Compact pruning")
        }
        return messages.map((message, index) => {
            const messageKey = message.id ?? `anonymous-${index}-${key(message.content)}`
            return {
                key: messageKey,
                // V2 request messages can revise a completed tool result in
                // place. Position-only stamps would replay a checkpoint over
                // different content with the same native message IDs.
                stamp: parseInt(key([message.role, message.content]).slice(0, 12), 16),
                // The core ladder has two virtual roles. A tool result is
                // generated input, not an assistant statement or human intent.
                role: message.role === "user" || message.role === "tool" ? "user" as const : "assistant" as const,
                ...(message.role === "tool" ? { prunableToolLike: true } : {}),
                handle: message,
                items: message.content.map((part, partIndex) => encodePart(part, messageKey, partIndex)),
            }
        })
    },
    decode(turns, messages) {
        return turns.flatMap((turn) => {
            const native = turn.handle as Message | undefined
            // Core can attach a prose stub to a pruned tool message. V2's
            // tool role accepts only tool-result parts: never serialize that
            // prose as a provider tool response. The original exact text is
            // in the archive/reference; an unmatched call fails validation.
            const content = turn.items.flatMap((item) => item.kind === "synthetic"
                ? native?.role === "tool" ? [] : [Message.text(item.text)]
                : [partOf(item)])
            // A pruned tool result with no remaining parts cannot be sent as
            // an empty tool response; the matching call is validated below.
            if (native && content.length === 0) return []
            if (native) return [Message.make({ ...native, content })]
            const base = messages.find((message) => message.role === "user")
            if (!base) throw new Error("Cannot insert a compacted handoff without a source user turn")
            return [Message.make({ role: "user", id: `bc-${turn.key}`, content, metadata: { betterCompact: "handoff", untrusted: true } })]
        })
    },
    estimateTurns(turns) {
        return countTokens(JSON.stringify(turns.map((turn) => ({ role: turn.role, content: turn.items.map((item) =>
            item.kind === "synthetic" ? { type: "text", text: item.text } : partOf(item),
        ) }))))
    },
    estimateItem(item) { return countTokens(JSON.stringify(item.handle)) },
    transcriptLine(item) { return item.kind === "synthetic" ? item.text : JSON.stringify(item.handle) },
    transcriptDocument(turns) {
        return JSON.stringify(turns.filter((turn) => turn.handle).map((turn) => turn.handle), null, 2)
    },
}

export const v2Conventions: Conventions = {
    isSkillItem(item) { return item.kind === "tool" && (item.handle as ContentPart).type === "tool-call" && (item.handle as Extract<ContentPart, { type: "tool-call" }>).name === "skill" },
    repeatableUserTextKey(text) {
        return text.startsWith("Continue working toward the active session goal.") && text.includes("<untrusted_objective>")
            ? "goal-continuation" : null
    },
    tool(item) {
        const part = item.handle as ContentPart
        if (part.type === "tool-call") return { name: part.name, input: part.input }
        if (part.type === "tool-result") return { name: part.name, input: undefined, error: part.result.type === "error" ? String(part.result.value) : undefined }
        return { name: "unknown", input: undefined }
    },
    isPreservedItem(item) {
        if (item.kind === "synthetic") return false
        const part = partOf(item)
        return part.type === "compaction" || (part.type === "reasoning" && !!part.encrypted)
    },
}

export const v2Spec: LadderSpec = {
    codec: v2Codec,
    conventions: v2Conventions,
    stages: [skillsStage, supersedeReadsStage, purgeErrorInputsStage, toolsOldStage, reasoningStage, toolsRemainingStage, assistantRunsStage],
}
