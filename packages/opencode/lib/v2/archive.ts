import { createHash } from "node:crypto"
import * as fs from "node:fs/promises"
import { join } from "node:path"
import type { Message } from "@opencode/ai"
import { ensurePrivateDirectory, readPrivateFile, writePrivateFile } from "../private-storage"
import { validateV2Description, validateV2Handoff } from "./handoff"
import { matchesMessageID } from "./identity"

const digest = (value: string) => createHash("sha256").update(value).digest("hex")
export const V2_ARCHIVE_RETENTION_MS = 7 * 24 * 60 * 60 * 1000
const sessionName = (id: string) => {
    if (!/^[a-zA-Z0-9_-]{1,160}$/.test(id)) throw new Error("Invalid archive session ID")
    return id
}
const base = (root: string, sessionID: string) => join(root, ".opencode", "better-compact", "v2", "sessions", sessionName(sessionID))
const catalogFile = (root: string, sessionID: string) => join(base(root, sessionID), "catalog.json")
const archiveFile = (root: string, sessionID: string, id: string) => {
    if (!/^c\d{6}-[a-f0-9]{12}$/.test(id)) throw new Error("Invalid archive ID")
    return join(base(root, sessionID), "archives", `${id}.json`)
}

async function withCatalogLock<T>(root: string, sessionID: string, work: () => Promise<T>): Promise<T> {
    const project = await fs.realpath(root)
    const directory = base(project, sessionID)
    await ensurePrivateDirectory(directory, project)
    const path = join(directory, ".catalog.lock")
    let handle: fs.FileHandle | undefined
    for (let attempt = 0; attempt < 150; attempt++) {
        try {
            handle = await fs.open(path, "wx", 0o600)
            break
        } catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error
            const info = await fs.lstat(path).catch((failure) => {
                if (failure instanceof Error && "code" in failure && failure.code === "ENOENT") return undefined
                throw failure
            })
            if (info && (!info.isFile() || info.isSymbolicLink())) throw new Error("Unsafe V2 catalog lock")
            if (info && Date.now() - info.mtimeMs > 60_000) {
                await fs.unlink(path)
                continue
            }
            await new Promise((resolve) => setTimeout(resolve, 20))
        }
    }
    if (!handle) throw new Error("V2 catalog is busy")
    try { return await work() }
    finally {
        await handle.close()
        await fs.unlink(path).catch((error) => {
            if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error
        })
    }
}

export interface V2ArchiveEntry {
    id: string
    sequence: number
    sha256: string
    messageIDs: string[]
    createdAt: string
    status: "pending" | "ready" | "expired"
    description?: string
    authorityKey?: string
}

export interface V2ArchiveCatalog {
    version: 2
    projectRoot: string
    sessionID: string
    entries: V2ArchiveEntry[]
    checkpoint?: {
        rangeHash: string
        archiveID: string
        handoff: string
        validatedAt: string
        authorityKey?: string
    }
    summaryAttempt?: { rangeHash: string; archiveID: string; reason: string; calls: number; at: string; authorityKey?: string }
    retirementThrough?: number
    /** Distinguish a prepared archive from a model-visible transform. */
    replayState?: "unapplied" | "applied"
    /** Latest validated replay boundary. Every later native message stays raw until a replacement handoff. */
    replayFrontier?: { anchors: Array<{ id: string; sha256: string }>; prefixSha256: string; authorityKey?: string; at: string }
}

/** Bind the whole native prefix, not just the last ID: earlier edits may leave
 * the last ID unchanged while invalidating an archived handoff. */
export function v2ReplayPrefixHash(messages: readonly Message[], originals: ReadonlyMap<string, Message>): string {
    return digest(JSON.stringify(messages.map((message) => originals.get(message.id!) ?? message)))
}

/** Commit the first successful replay boundary before mutating the outgoing
 * request. Only a validated replacement handoff may advance it later. */
export async function saveV2ReplayFrontier(root: string, sessionID: string, messages: readonly Message[],
    originals: ReadonlyMap<string, Message>, advance = false, authorityKey?: string): Promise<void> {
    return withCatalogLock(root, sessionID, async () => {
        const catalog = await loadV2Catalog(root, sessionID)
        if (catalog.replayFrontier && !advance) return
        if (!catalog.entries.some((entry) => entry.status !== "expired" && entry.authorityKey === authorityKey))
            throw new Error("Missing exact V2 archive for replay")
        const anchors = messages.slice(-8).map((message) => {
            if (!message.id || message.id.length > 512) throw new Error("Invalid replay message identity")
            return { id: message.id, sha256: digest(JSON.stringify(originals.get(message.id) ?? message)) }
        })
        if (!anchors.length) throw new Error("Empty replay boundary")
        catalog.replayFrontier = { anchors, prefixSha256: v2ReplayPrefixHash(messages, originals),
            authorityKey, at: new Date().toISOString() }
        catalog.replayState = "applied"
        await writePrivateFile(catalogFile(catalog.projectRoot, sessionID), JSON.stringify(catalog), catalog.projectRoot)
    })
}

export async function loadV2Catalog(root: string, sessionID: string): Promise<V2ArchiveCatalog> {
    const projectRoot = await fs.realpath(root)
    let serialized: string
    try {
        serialized = await readPrivateFile(catalogFile(projectRoot, sessionID), projectRoot)
    } catch (error) {
        if (error instanceof Error && "code" in error && error.code === "ENOENT")
            return { version: 2, projectRoot, sessionID, entries: [] }
        throw error
    }
    const catalog: unknown = JSON.parse(serialized)
    if (!catalog || typeof catalog !== "object" || (catalog as V2ArchiveCatalog).version !== 2 ||
        (catalog as V2ArchiveCatalog).projectRoot !== projectRoot ||
        (catalog as V2ArchiveCatalog).sessionID !== sessionID ||
        !Array.isArray((catalog as V2ArchiveCatalog).entries))
        throw new Error("Invalid or foreign V2 archive catalog")
    return catalog as V2ArchiveCatalog
}

/** Never trust an ID or a path supplied by a model without a matching catalog entry. */
export async function readV2Archive(root: string, sessionID: string, id: string): Promise<readonly Message[]> {
    const catalog = await loadV2Catalog(root, sessionID)
    const entry = catalog.entries.find((candidate) => candidate.id === id && candidate.status !== "expired")
    if (!entry) throw new Error("Unknown or expired archive")
    const text = await readPrivateFile(archiveFile(catalog.projectRoot, sessionID, entry.id), catalog.projectRoot)
    if (digest(text) !== entry.sha256) throw new Error("Modified archive")
    const messages: unknown = JSON.parse(text)
    if (!Array.isArray(messages) || !messages.every((item, index) =>
        item && typeof item === "object" && matchesMessageID(item as Message, entry.messageIDs[index]!)) ||
        messages.length !== entry.messageIDs.length) throw new Error("Invalid archive payload")
    return messages as Message[]
}

/** Only byte-identical, older, ready human messages may leave a validated second checkpoint. */
export async function retiredV2HumanIDs(root: string, sessionID: string, source: readonly Message[],
    through?: number, validatedHandoff?: string, originals: ReadonlyMap<string, Message> = new Map(),
    authorityKey?: string): Promise<Set<string>> {
    const ids = new Set<string>()
    if (!through) return ids
    const catalog = await loadV2Catalog(root, sessionID)
    const proposed = catalog.checkpoint?.authorityKey === authorityKey &&
        catalog.entries.some((item) => item.sequence === through + 1 && item.authorityKey === authorityKey) &&
        catalog.entries.filter((item) => item.authorityKey === authorityKey && item.sequence <= through)
            .every((item) => item.status === "ready" && !!item.description)
    if (through > (catalog.retirementThrough ?? 0) && !proposed)
        throw new Error("Unvalidated archive retirement boundary")
    const current = new Map(source.filter((message) => message.role === "user" && message.id)
        .map((message) => [message.id!, JSON.stringify(originals.get(message.id!) ?? message)]))
    const normalize = (text: string) => text.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim().replace(/\s+/g, " ")
    const handoff = normalize(validatedHandoff ?? "")
    const cohort = catalog.entries.filter((item) => item.authorityKey === authorityKey && item.sequence <= through)
    if (cohort.some((entry) => entry.status !== "ready" || !entry.description)) return ids
    for (const entry of cohort) {
        for (const [index, message] of (await readV2Archive(root, sessionID, entry.id)).entries()) {
            const id = entry.messageIDs[index]!
            if (message.role !== "user" || current.get(id) !== JSON.stringify(message)) continue
            const wording = message.content.filter((part) => part.type === "text").map((part) => normalize(part.text))
            if (wording.length && wording.every((text) => !!text && handoff.includes(text))) ids.add(id)
        }
    }
    return ids
}

/** Archive exact original message values, never a pruned or synthetic request. */
export async function appendV2Archive(root: string, sessionID: string, messages: readonly Message[],
    originals: ReadonlyMap<string, Message> = new Map(), authorityKey?: string): Promise<V2ArchiveEntry | null> {
    return withCatalogLock(root, sessionID, async () => {
    const catalog = await loadV2Catalog(root, sessionID)
    if (!catalog.entries.length) catalog.replayState = "unapplied"
    const covered = new Set<string>()
    for (const entry of catalog.entries) {
        if (entry.status === "expired" || entry.authorityKey !== authorityKey) continue
        // Validate old files before relying on coverage: do not silently skip
        // history if an archive was removed or modified outside the plugin.
        const old = await readV2Archive(catalog.projectRoot, sessionID, entry.id)
        old.forEach((message, index) => covered.add(`${entry.messageIDs[index]}:${digest(JSON.stringify(message))}`))
    }
    const delta = messages.filter((message) => {
        if (!message.id) throw new Error("Cannot archive a native message without ID")
        const raw = originals.get(message.id) ?? message
        if (!matchesMessageID(raw, message.id)) throw new Error("Inconsistent V2 archive identity")
        return !covered.has(`${message.id}:${digest(JSON.stringify(raw))}`)
    })
    if (!delta.length) return null
    const sequence = (catalog.entries.at(-1)?.sequence ?? 0) + 1
    const text = JSON.stringify(delta.map((message) => originals.get(message.id!) ?? message))
    const sha256 = digest(text)
    const id = `c${String(sequence).padStart(6, "0")}-${sha256.slice(0, 12)}`
    const entry: V2ArchiveEntry = {
        id, sequence, sha256, messageIDs: delta.map((message) => message.id!),
        createdAt: new Date().toISOString(), status: "pending", authorityKey,
    }
    await writePrivateFile(join(catalog.projectRoot, ".opencode", "better-compact", ".gitignore"), "*\n!.gitignore\n", catalog.projectRoot)
    const path = archiveFile(catalog.projectRoot, sessionID, id)
    try {
        // A crash between the raw write and catalog write may leave this
        // deterministic orphan. Reuse only identical bytes; never overwrite
        // unexpected data at an existing private archive path.
        const existing = await readPrivateFile(path, catalog.projectRoot)
        if (existing !== text) throw new Error("Conflicting archive at expected ID")
    } catch (error) {
        if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error
        await writePrivateFile(path, text, catalog.projectRoot)
    }
    const verified = await readPrivateFile(path, catalog.projectRoot)
    if (digest(verified) !== sha256) throw new Error("Archive verification failed")
    catalog.entries.push(entry)
    await writePrivateFile(catalogFile(catalog.projectRoot, sessionID), JSON.stringify(catalog), catalog.projectRoot)
    return entry
    })
}

/** Publish validated checkpoint and description together; replay cannot publish an unverified model draft. */
export async function saveV2Checkpoint(root: string, sessionID: string, input: {
    rangeHash: string; archiveID: string; handoff: string; description: string; recentUserIntent: string[]; authorityKey?: string
}): Promise<void> {
    return withCatalogLock(root, sessionID, async () => {
    if (!/^[a-f0-9]{16}$/.test(input.rangeHash)) throw new Error("Invalid checkpoint range hash")
    const catalog = await loadV2Catalog(root, sessionID)
    if (validateV2Handoff(JSON.stringify({ handoff: input.handoff }), input.recentUserIntent,
        catalog.checkpoint?.authorityKey === input.authorityKey ? catalog.checkpoint?.handoff : undefined) !== input.handoff ||
        validateV2Description(input.description, input.archiveID) !== input.description)
        throw new Error("Unvalidated checkpoint or description")
    const entry = catalog.entries.find((item) => item.id === input.archiveID && item.status !== "expired")
    if (!entry || entry.authorityKey !== input.authorityKey) throw new Error("Missing archive for checkpoint")
    await readV2Archive(root, sessionID, entry.id)
    if (catalog.checkpoint?.rangeHash === input.rangeHash && catalog.checkpoint.authorityKey === input.authorityKey) {
        if (catalog.checkpoint.handoff !== input.handoff || catalog.checkpoint.archiveID !== entry.id)
            throw new Error("Conflicting checkpoint for this range")
        return
    }
    entry.description = input.description
    entry.status = "ready"
    if (catalog.checkpoint?.authorityKey === input.authorityKey && entry.sequence >= 2 &&
        catalog.entries.some((item) => item.authorityKey === input.authorityKey && item.sequence < entry.sequence) &&
        catalog.entries
        .filter((item) => item.authorityKey === input.authorityKey && item.sequence < entry.sequence)
        .every((item) => item.status === "ready" && !!item.description))
        catalog.retirementThrough = entry.sequence - 1
    if (catalog.checkpoint?.authorityKey !== input.authorityKey) delete catalog.retirementThrough
    catalog.checkpoint = { rangeHash: input.rangeHash, archiveID: entry.id, handoff: input.handoff,
        authorityKey: input.authorityKey, validatedAt: new Date().toISOString() }
    await writePrivateFile(catalogFile(catalog.projectRoot, sessionID), JSON.stringify(catalog), catalog.projectRoot)
    })
}

/** Failed model calls are not retried for the same unchanged boundary. */
export async function recordV2SummaryFailure(root: string, sessionID: string, input: {
    rangeHash: string; archiveID: string; reason: string; calls: number; authorityKey?: string
}): Promise<void> {
    return withCatalogLock(root, sessionID, async () => {
    if (!/^[a-f0-9]{16}$/.test(input.rangeHash) ||
        !["input_does_not_fit", "transport_error", "invalid_output", "valid_but_not_smaller"].includes(input.reason) ||
        !Number.isSafeInteger(input.calls) || input.calls < 0 || input.calls > 7)
        throw new Error("Invalid summary failure metadata")
    const catalog = await loadV2Catalog(root, sessionID)
    if (!catalog.entries.some((entry) => entry.id === input.archiveID && entry.status !== "expired" &&
        entry.authorityKey === input.authorityKey))
        throw new Error("Missing archive for summary failure")
    catalog.summaryAttempt = { ...input, at: new Date().toISOString() }
    await writePrivateFile(catalogFile(catalog.projectRoot, sessionID), JSON.stringify(catalog), catalog.projectRoot)
    })
}

/** Expire only this session's catalog-authorized raw files; retain ID tombstones. */
export async function expireV2Archives(root: string, sessionID: string, now = Date.now()): Promise<number> {
    return withCatalogLock(root, sessionID, async () => {
    const catalog = await loadV2Catalog(root, sessionID)
    const expired = catalog.entries.filter((entry) => entry.status !== "expired" &&
        Number.isFinite(Date.parse(entry.createdAt)) &&
        Date.parse(entry.createdAt) <= now - V2_ARCHIVE_RETENTION_MS)
    // Verify *all* candidates first; a foreign/symlinked/changed file must
    // not be silently removed by housekeeping.
    for (const entry of expired) await readV2Archive(catalog.projectRoot, sessionID, entry.id)
    if (expired.length) {
        for (const entry of expired) entry.status = "expired"
        await writePrivateFile(catalogFile(catalog.projectRoot, sessionID), JSON.stringify(catalog), catalog.projectRoot)
    }
    // If a prior process crashed after publishing tombstones, finish its
    // deletion on the next activity. Never unlink an unverified/symlink file.
    for (const entry of catalog.entries.filter((item) => item.status === "expired")) {
        const path = archiveFile(catalog.projectRoot, sessionID, entry.id)
        let bytes: string
        try { bytes = await readPrivateFile(path, catalog.projectRoot) }
        catch (error) {
            if (error instanceof Error && "code" in error && error.code === "ENOENT") continue
            throw error
        }
        if (digest(bytes) !== entry.sha256) throw new Error("Modified expired archive")
        await fs.unlink(path)
    }
    return expired.length
    })
}
