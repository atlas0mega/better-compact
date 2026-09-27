import type { Message } from "@opencode/ai"

/** A request transform is optional; an invalid transform must never corrupt a provider request. */
export function validateRequestTransform(before: readonly Message[], after: readonly Message[], options: {
    /** Only an independently validated, exact, session-owned archive permits retiring old human wording. */
    archivedHumanIds?: ReadonlySet<string>
} = {}): void {
    const invariant = (condition: boolean, reason: string) => {
        if (!condition) throw new Error(`Unsafe Better Compact V2 transform: ${reason}`)
    }
    invariant(before.every((message) => !!message.id), "native message without stable ID")
    const lastSystem = before.findLastIndex((message) => message.role === "system")
    const systems = before.filter((message) => message.role === "system")
    const emittedSystems = after.filter((message) => message.role === "system")
    invariant(emittedSystems.every((message) => systems.includes(message)), "new system message")
    invariant(emittedSystems.length === systems.length && emittedSystems.every((message, index) =>
        message === systems[index]), "system authority prefix changed")
    if (lastSystem >= 0) invariant(before.slice(0, lastSystem + 1).every((message, index) =>
        after[index] === message), "system authority prefix changed")
    const original = new Map(before.map((message) => [message.id!, message]))
    const emitted = new Set<string>()
    const calls = new Map<string, number>()
    const results = new Map<string, number>()
    for (const message of after) {
        invariant(message.content.length > 0, "empty message")
        if (message.role === "system") {
            invariant(before.some((source) => source === message), "new system message")
            continue
        }
        const source = message.id ? original.get(message.id) : undefined
        if (source) {
            invariant(!emitted.has(message.id!), "duplicate native message")
            invariant(source.role === message.role, "native role changed")
            emitted.add(message.id!)
        } else {
            invariant(message.role === "user" && message.metadata?.betterCompact === "handoff" && message.metadata?.untrusted === true,
                "untrusted synthetic content without provenance")
        }
        for (const part of message.content) {
            if (part.type === "tool-call") calls.set(part.id, (calls.get(part.id) ?? 0) + 1)
            if (part.type === "tool-result") results.set(part.id, (results.get(part.id) ?? 0) + 1)
        }
        if (message.role === "tool") invariant(message.content.every((part) => part.type === "tool-result"), "malformed tool result")
    }
    // Existing partial history may already have unmatched calls. The plugin
    // cannot introduce new unmatched calls or results by pruning one side.
    const beforeCalls = new Map<string, number>()
    const beforeResults = new Map<string, number>()
    for (const message of before) for (const part of message.content) {
        if (part.type === "tool-call") beforeCalls.set(part.id, (beforeCalls.get(part.id) ?? 0) + 1)
        if (part.type === "tool-result") beforeResults.set(part.id, (beforeResults.get(part.id) ?? 0) + 1)
    }
    for (const id of new Set([...beforeCalls.keys(), ...beforeResults.keys(), ...calls.keys(), ...results.keys()])) {
        const originalDifference = (beforeCalls.get(id) ?? 0) - (beforeResults.get(id) ?? 0)
        invariant((calls.get(id) ?? 0) - (results.get(id) ?? 0) === originalDifference, `orphaned tool pair ${id}`)
    }
    for (const message of before) {
        if (message.role === "user" && !options.archivedHumanIds?.has(message.id!)) {
            const retained = after.find((candidate) => candidate.id === message.id)
            invariant(retained !== undefined && JSON.stringify(retained.content) === JSON.stringify(message.content),
                `human instruction removed or rewritten: ${message.id}`)
        }
        for (const part of message.content) {
            if (part.type !== "compaction" && part.type !== "media" && part.type !== "effort" &&
                !(part.type === "reasoning" && part.encrypted)) continue
            invariant(after.some((candidate) => candidate.id === message.id && candidate.content.some((current) =>
                JSON.stringify(current) === JSON.stringify(part))), `opaque provider state removed: ${message.id}`)
        }
    }
    const latestHuman = before.findLast((message) => message.role === "user")
    if (latestHuman) invariant(after.findLast((message) => message.role === "user" && !message.metadata?.betterCompact)?.id === latestHuman.id,
        "current human turn displaced")
}
