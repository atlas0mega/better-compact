import { createHash } from "node:crypto"
import { Message } from "@opencode/ai"

/** V2 provider-request messages commonly have no ID. Keep a private, deterministic
 * identity for planning/validation; never add that identity to the exact archive. */
export function virtualMessageID(message: Message, index: number): string {
    const hash = createHash("sha256").update(JSON.stringify([index, message])).digest("hex").slice(0, 16)
    return `bcv-${index}-${hash}`
}

export function matchesMessageID(message: Message, id: string): boolean {
    if (message.id) return message.id === id
    const match = /^bcv-(0|[1-9]\d*)-[a-f0-9]{16}$/.exec(id)
    return !!match && Number.isSafeInteger(Number(match[1])) && virtualMessageID(message, Number(match[1])) === id
}

export function identifyV2Messages(original: readonly Message[]): {
    messages: Message[]
    originals: ReadonlyMap<string, Message>
} {
    const originals = new Map<string, Message>()
    const messages = original.map((message, index) => {
        const id = message.id ?? virtualMessageID(message, index)
        if (originals.has(id)) throw new Error("Duplicate V2 native message identity")
        originals.set(id, message)
        return message.id ? message : Message.make({ ...message, id })
    })
    return { messages, originals }
}

/** Return unmodified original instances whenever possible, and remove only the
 * private identity from modified ID-less messages before sending to the model. */
export function restoreV2Messages(messages: readonly Message[], originals: ReadonlyMap<string, Message>): Message[] {
    return messages.map((message) => {
        const original = message.id ? originals.get(message.id) : undefined
        if (!original) return message
        if (JSON.stringify(message.content) === JSON.stringify(original.content)) return original
        if (original.id) return message
        const { id: _privateID, ...content } = message
        return Message.make(content)
    })
}
