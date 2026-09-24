import { createHash } from "node:crypto"
import type { WithParts } from "../state"

export const PREFIX_FINGERPRINT_VERSION = 2

/** Old snapshots retain the exact original hash until the next real compaction. */
export function boundaryRangeHash(messages: WithParts[], version: 1 | 2 = 2): string {
    const seed = JSON.stringify(
        messages.map((message) => ({
            info:
                version === 1
                    ? withoutTransportFields(message.info, ["id", "sessionID", "parentID"])
                    : semanticMessageInfo(message.info),
            parts: message.parts.map((part) =>
                version === 1
                    ? withoutTransportFields(part, ["id", "messageID", "sessionID"])
                    : semanticPart(part),
            ),
        })),
    )
    return createHash("sha256").update(seed).digest("hex").slice(0, 16)
}

export function boundarySnapshotHash(
    messages: WithParts[],
    snapshot: { prefixFingerprintVersion?: number },
): string {
    return boundaryRangeHash(messages, snapshot.prefixFingerprintVersion === 2 ? 2 : 1)
}

function semanticMessageInfo(info: WithParts["info"]): Record<string, unknown> {
    const result = withoutTransportFields(info, [
        "id",
        "sessionID",
        "parentID",
        "tokens",
        "cost",
        "finish",
        "mode",
        "path",
        "time",
    ])
    if (info.time?.created !== undefined) result.time = { created: info.time.created }
    return result
}

function semanticPart(part: WithParts["parts"][number]): Record<string, unknown> {
    const result = withoutTransportFields(part, ["id", "messageID", "sessionID", "time"])
    if (part.type === "tool" && part.state) {
        result.state = withoutTransportFields(part.state, ["time"])
    }
    return result
}

function withoutTransportFields(value: object, fields: string[]): Record<string, unknown> {
    const result = { ...value } as Record<string, unknown>
    for (const field of fields) delete result[field]
    return result
}
