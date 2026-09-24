import type { WithParts } from "../state"

// The Syndicate Engine plugins' shared buildInjectionBody appends this suffix
// to their final text part. OpenCode persists these prompts with role=user,
// even though they are generated context rather than human instructions.
const PROVENANCE_SUFFIX = /\n\n\[plugin-injection:[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}\]$/

export function isSyndicatePluginInjection(message: WithParts): boolean {
    if (message.info.role !== "user" || message.parts.length === 0) return false
    if (
        !message.parts.every(
            (part) => part.type === "text" && !("ignored" in part && part.ignored === true),
        )
    )
        return false
    const last = message.parts.at(-1)
    return last?.type === "text" && PROVENANCE_SUFFIX.test(last.text)
}

/** Goal auto-continuations carry the current task state. Exclude them from
 * human-message budgets and tail boundaries, but keep the latest in the live
 * handoff rather than treating it as a disposable tool result. */
export function isGoalPluginStatePrompt(message: WithParts): boolean {
    if (message.info.role !== "user" || message.parts.length !== 1) return false
    const part = message.parts[0]
    if (part.type !== "text" || part.ignored) return false
    return (
        ((part.text.startsWith("Continue working toward the active session goal.\n\n") &&
            part.text.includes("\nBudget:\n")) ||
            (part.text.startsWith("The active session goal has reached a safety limit.\n\n") &&
                part.text.includes("\nStop reason: "))) &&
        /<untrusted_objective>\n[\s\S]*?\n<\/untrusted_objective>/.test(part.text)
    )
}

/** One entry point for generated user-role plugin messages. The ZIP's UUID
 * messages are tool-like; goal-state prompts use a distinct protected lane. */
export function isPluginGeneratedUserMessage(message: WithParts): boolean {
    return isSyndicatePluginInjection(message) || isGoalPluginStatePrompt(message)
}
