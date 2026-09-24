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
