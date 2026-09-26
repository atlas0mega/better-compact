import { createHash } from "node:crypto"
import * as fs from "node:fs/promises"
import { dirname, isAbsolute, join, relative, resolve, sep } from "node:path"
import type { BoundaryContextPlan } from "@better-compact/core"
import type { WithParts } from "../state"
import { readPrivateFile, writePrivateFile } from "../private-storage"
import { safePathPart } from "./transcripts"

export interface ArchiveEntry {
    id: string
    sequence: number
    relativePath: string
    rangeHash: string
    firstMessageId: string
    lastMessageId: string
    fingerprints: Record<string, string>
    checksum: string
    createdAt: string
    status: "pending" | "ready" | "expired"
    /** Non-sensitive last failure while awaiting a validated handoff. */
    failureReason?: string
    descriptionAttempts?: number
    description?: string
    /** A rejected model summary is evidence, never automatically live context. */
    oversizedSummaryPath?: string
    oversizedSummaryChecksum?: string
    oversizedSummaryArchiveIds?: string[]
}

export interface ArchiveCatalog {
    version: 1
    ownerSessionId: string
    projectRoot: string
    nextSequence: number
    entries: ArchiveEntry[]
    validatedCheckpointId?: string
    checkpoint?: string
    retirementThrough?: number
    inheritedLinks?: Array<{ ownerSessionId: string; prefixFingerprint: string }>
}

export function liveArchiveDescriptions(catalog: ArchiveCatalog): string {
    return catalog.entries
        .filter((entry) => entry.status === "ready" && entry.description && !archiveExpired(entry))
        .slice(-5)
        .map((entry) => `- ${entry.id} — ${entry.description!.replace(/\s+/g, " ").trim()}`)
        .join("\n")
}

/** Only older described deltas can age out of live wording at a later validated boundary. */
export function eligibleRetirementThrough(
    catalog: ArchiveCatalog,
    currentBoundarySequence: number,
): number | undefined {
    let through = catalog.retirementThrough ?? 0
    for (const entry of catalog.entries) {
        if (entry.sequence <= through) continue
        if (entry.sequence >= currentBoundarySequence) break
        if (entry.sequence !== through + 1 || entry.status !== "ready" || !entry.description)
            break
        through = entry.sequence
    }
    return through || undefined
}

const rootName = ".opencode/better-compact"
export const ARCHIVE_RETENTION_MS = 7 * 24 * 60 * 60 * 1000
const sha = (text: string) => createHash("sha256").update(text).digest("hex")
export const nativeMessageFingerprint = (message: WithParts): string =>
    sha(JSON.stringify(message))

export function archiveExpired(entry: ArchiveEntry, now = Date.now()): boolean {
    const created = Date.parse(entry.createdAt)
    return (
        entry.status === "expired" ||
        (Number.isFinite(created) && created <= now - ARCHIVE_RETENTION_MS)
    )
}

function validSession(id: string): string {
    if (!/^[a-zA-Z0-9_-]{1,160}$/.test(id)) throw new Error("Invalid archive session ID")
    return id
}

function sessionRoot(directory: string, sessionId: string): string {
    return join(directory, rootName, "sessions", validSession(sessionId))
}

function catalogPath(directory: string, sessionId: string): string {
    return join(sessionRoot(directory, sessionId), "catalog.json")
}

export function archiveCatalogPath(directory: string, sessionId: string): string {
    return catalogPath(directory, sessionId)
}

export function archiveRelativePath(sessionId: string, id: string): string {
    if (!/^c\d{6}-[a-f0-9]{12}$/.test(id)) throw new Error("Invalid archive ID")
    return `${rootName}/sessions/${safePathPart(validSession(sessionId))}/archives/${id}.md`
}

/** Catalog-authorized paths only. Never accept a model-supplied path. */
export function authorizedArchivePath(
    directory: string,
    sessionId: string,
    entry: ArchiveEntry,
): string {
    if (entry.relativePath !== archiveRelativePath(sessionId, entry.id)) {
        throw new Error("Archive path does not match its catalog ID")
    }
    const full = resolve(directory, entry.relativePath)
    const within = relative(resolve(directory), full)
    if (within === ".." || within.startsWith(`..${sep}`) || isAbsolute(within)) {
        throw new Error("Archive path escapes project")
    }
    return full
}

export async function loadArchiveCatalog(
    directory: string,
    sessionId: string,
): Promise<ArchiveCatalog> {
    const projectRoot = await fs.realpath(directory)
    const path = catalogPath(directory, sessionId)
    let content: string
    try {
        content = await readPrivateFile(path, directory)
    } catch (error) {
        if (error instanceof Error && "code" in error && error.code === "ENOENT") {
            return {
                version: 1,
                ownerSessionId: sessionId,
                projectRoot,
                nextSequence: 1,
                entries: [],
            }
        }
        throw error
    }
    const catalog: unknown = JSON.parse(content)
    if (
        !catalog ||
        typeof catalog !== "object" ||
        (catalog as ArchiveCatalog).version !== 1 ||
        (catalog as ArchiveCatalog).ownerSessionId !== sessionId ||
        (catalog as ArchiveCatalog).projectRoot !== projectRoot ||
        !Array.isArray((catalog as ArchiveCatalog).entries) ||
        !Number.isSafeInteger((catalog as ArchiveCatalog).nextSequence)
    )
        throw new Error("Invalid or foreign Better Compact archive catalog")
    return catalog as ArchiveCatalog
}

/** Only catalogs under this project's private session root are startup candidates. */
export async function pendingArchiveSessionIds(directory: string): Promise<string[]> {
    let entries: import("node:fs").Dirent[]
    try {
        entries = await fs.readdir(join(directory, rootName, "sessions"), { withFileTypes: true })
    } catch (error) {
        if (error instanceof Error && "code" in error && error.code === "ENOENT") return []
        throw error
    }
    const pending: string[] = []
    for (const entry of entries) {
        if (!entry.isDirectory() || !/^[a-zA-Z0-9_-]{1,160}$/.test(entry.name)) continue
        const catalog = await expireArchives(directory, entry.name)
        if (
            catalog.entries.some(
                (item) => item.status === "pending" && (item.descriptionAttempts ?? 0) < 5,
            )
        )
            pending.push(entry.name)
    }
    return pending
}

/** Prevent separate OpenCode processes from billing the same session backlog. */
export async function tryArchiveDescriptionLock(
    directory: string,
    sessionId: string,
): Promise<(() => Promise<void>) | null> {
    const path = join(dirname(catalogPath(directory, sessionId)), ".description.lock")
    const parent = dirname(path)
    const realParent = await fs.realpath(parent)
    if (realParent !== resolve(parent)) throw new Error("Unsafe archive lock directory")
    for (let attempt = 0; attempt < 2; attempt++) {
        try {
            const handle = await fs.open(path, "wx", 0o600)
            return async () => {
                await handle.close()
                await fs.unlink(path).catch((error) => {
                    if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
                        throw error
                })
            }
        } catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error
            const info = await fs.lstat(path)
            if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe archive lock")
            if (Date.now() - info.mtimeMs < 60 * 60 * 1000) return null
            await fs.unlink(path)
        }
    }
    return null
}

/** Backfill only the opened project; never walk its unrelated descendants. */
export async function projectArchiveRoots(cwd: string): Promise<string[]> {
    const root = await fs.realpath(cwd)
    try {
        for (const component of [".opencode", rootName, join(rootName, "sessions")]) {
            const info = await fs.lstat(join(root, component))
            if (info.isSymbolicLink()) throw new Error("Unsafe archive startup directory")
            if (!info.isDirectory()) return []
        }
        return [root]
    } catch (error) {
        if (error instanceof Error && "code" in error && error.code === "ENOENT") return []
        throw error
    }
}

export async function saveArchiveCatalog(
    directory: string,
    catalog: ArchiveCatalog,
): Promise<void> {
    const current = await fs.realpath(directory)
    if (catalog.projectRoot !== current)
        throw new Error("Archive catalog belongs to another project")
    await writePrivateFile(join(directory, rootName, ".gitignore"), "*\n!.gitignore\n", directory)
    await withCatalogLock(directory, catalog.ownerSessionId, () =>
        persistCatalogUnderLock(directory, catalog),
    )
}

async function withCatalogLock<T>(
    directory: string,
    sessionId: string,
    work: () => Promise<T>,
): Promise<T> {
    const path = join(directory, rootName, `.catalog-${validSession(sessionId)}.lock`)
    const parent = dirname(path)
    if ((await fs.realpath(parent)) !== resolve(parent))
        throw new Error("Unsafe catalog lock directory")
    for (let attempt = 0; attempt < 100; attempt++) {
        let handle: import("node:fs/promises").FileHandle
        try {
            handle = await fs.open(path, "wx", 0o600)
        } catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "EEXIST")) throw error
            const info = await fs.lstat(path).catch((readError) => {
                if (
                    readError instanceof Error &&
                    "code" in readError &&
                    readError.code === "ENOENT"
                )
                    return null
                throw readError
            })
            if (info && (!info.isFile() || info.isSymbolicLink()))
                throw new Error("Unsafe catalog lock")
            if (info && Date.now() - info.mtimeMs > 60_000) await fs.unlink(path).catch(() => {})
            await new Promise((resolve) => setTimeout(resolve, 20))
            continue
        }
        try {
            return await work()
        } finally {
            await handle.close()
            await fs.unlink(path)
        }
    }
    throw new Error("Archive catalog write lock timed out")
}

function mergeArchiveCatalog(latest: ArchiveCatalog, proposed: ArchiveCatalog): ArchiveCatalog {
    const entries = new Map(latest.entries.map((entry) => [entry.id, entry]))
    for (const item of proposed.entries) {
        const current = entries.get(item.id)
        if (!current) {
            entries.set(item.id, item)
            continue
        }
        const merged = { ...current, ...item }
        merged.descriptionAttempts = Math.max(
            current.descriptionAttempts ?? 0,
            item.descriptionAttempts ?? 0,
        )
        if (current.status === "expired" || item.status === "expired") {
            merged.status = "expired"
            delete merged.description
        } else if (current.status === "ready" || item.status === "ready") {
            merged.status = "ready"
            merged.description = item.description ?? current.description
            delete merged.failureReason
        }
        entries.set(item.id, merged)
    }
    const ordered = [...entries.values()].sort((a, b) => a.sequence - b.sequence)
    const selected = [latest, proposed]
        .sort(
            (a, b) =>
                (a.entries.find((entry) => entry.id === a.validatedCheckpointId)?.sequence ?? 0) -
                (b.entries.find((entry) => entry.id === b.validatedCheckpointId)?.sequence ?? 0),
        )
        .at(-1)!
    return {
        ...latest,
        ...proposed,
        entries: ordered,
        nextSequence: Math.max(
            latest.nextSequence,
            proposed.nextSequence,
            ...ordered.map((entry) => entry.sequence + 1),
        ),
        checkpoint: selected.checkpoint,
        validatedCheckpointId: selected.validatedCheckpointId,
        retirementThrough:
            Math.max(latest.retirementThrough ?? 0, proposed.retirementThrough ?? 0) || undefined,
        inheritedLinks: [
            ...(latest.inheritedLinks ?? []),
            ...(proposed.inheritedLinks ?? []),
        ].filter(
            (link, index, links) =>
                links.findIndex(
                    (other) =>
                        other.ownerSessionId === link.ownerSessionId &&
                        other.prefixFingerprint === link.prefixFingerprint,
                ) === index,
        ),
    }
}

async function persistCatalogUnderLock(directory: string, catalog: ArchiveCatalog): Promise<void> {
    const latest = await loadArchiveCatalog(directory, catalog.ownerSessionId)
    const merged = mergeArchiveCatalog(latest, catalog)
    await writePrivateFile(
        catalogPath(directory, catalog.ownerSessionId),
        JSON.stringify(merged, null, 2),
        directory,
    )
    Object.assign(catalog, merged)
}

/** Expire session-private payloads on activity; preserve ID/fingerprint tombstones. */
export async function expireArchives(
    directory: string,
    sessionId: string,
    now = Date.now(),
): Promise<ArchiveCatalog> {
    const catalog = await loadArchiveCatalog(directory, sessionId)
    const expired = catalog.entries.filter((entry) => archiveExpired(entry, now))
    if (!expired.length) return catalog
    if (expired.some((entry) => entry.status !== "expired")) {
        for (const entry of expired) {
            entry.status = "expired"
            delete entry.description
            delete entry.failureReason
        }
        // Readers see tombstones before any payload is removed.
        await saveArchiveCatalog(directory, catalog)
    }
    for (const entry of expired) {
        const paths = [authorizedArchivePath(directory, sessionId, entry)]
        if (entry.oversizedSummaryPath)
            paths.push(authorizedArchivedSummaryPath(directory, catalog, entry.id))
        for (const path of paths) {
            try {
                const info = await fs.lstat(path)
                if (!info.isFile() || info.isSymbolicLink()) throw new Error("Unsafe archive file")
                await fs.unlink(path)
            } catch (error) {
                if (!(error instanceof Error && "code" in error && error.code === "ENOENT"))
                    throw error
            }
        }
        delete entry.oversizedSummaryPath
        delete entry.oversizedSummaryChecksum
        delete entry.oversizedSummaryArchiveIds
    }
    await saveArchiveCatalog(directory, catalog)
    return catalog
}

export async function recordArchiveFailure(
    directory: string,
    catalog: ArchiveCatalog,
    reason: string,
): Promise<void> {
    const latest = await loadArchiveCatalog(directory, catalog.ownerSessionId)
    for (const entry of latest.entries) {
        if (entry.status === "pending") entry.failureReason = reason
    }
    await saveArchiveCatalog(directory, latest)
}

/** Called only after the fork's actual prefix passes the content-fingerprint check. */
export async function inheritArchiveCatalog(
    directory: string,
    forkSessionId: string,
    ownerSessionId: string,
    prefixFingerprint: string,
): Promise<void> {
    if (forkSessionId === ownerSessionId || !/^[a-f0-9]{16}$/.test(prefixFingerprint)) return
    const owner = await loadArchiveCatalog(directory, ownerSessionId)
    if (owner.entries.length === 0) return
    const fork = await loadArchiveCatalog(directory, forkSessionId)
    if (
        fork.inheritedLinks?.some(
            (link) =>
                link.ownerSessionId === ownerSessionId &&
                link.prefixFingerprint === prefixFingerprint,
        )
    )
        return
    fork.inheritedLinks = [...(fork.inheritedLinks ?? []), { ownerSessionId, prefixFingerprint }]
    await saveArchiveCatalog(directory, fork)
}

/** Return the newly covered native messages, including changed versions of old IDs. */
export function uncoveredMessages(messages: WithParts[], catalog: ArchiveCatalog): WithParts[] {
    const covered = new Map<string, Set<string>>()
    for (const entry of catalog.entries) {
        for (const [id, fingerprint] of Object.entries(entry.fingerprints)) {
            const revisions = covered.get(id) ?? new Set<string>()
            revisions.add(fingerprint)
            covered.set(id, revisions)
        }
    }
    return messages.filter(
        (message) => !covered.get(message.info.id)?.has(nativeMessageFingerprint(message)),
    )
}

export async function archiveBoundaryDelta(input: {
    directory: string
    sessionId: string
    plan: BoundaryContextPlan
    originalMessages: WithParts[]
}): Promise<{ catalog: ArchiveCatalog; entry: ArchiveEntry | null }> {
    await writePrivateFile(
        join(input.directory, rootName, ".gitignore"),
        "*\n!.gitignore\n",
        input.directory,
    )
    return withCatalogLock(input.directory, input.sessionId, async () => {
        const catalog = await loadArchiveCatalog(input.directory, input.sessionId)
        const ids = new Set(input.plan.transcript.messageIds)
        const eligible = input.originalMessages.filter((message) => ids.has(message.info.id))
        const delta = uncoveredMessages(eligible, catalog)
        if (delta.length === 0) return { catalog, entry: null }
        const content = [
            "# Better Compact Delta Archive",
            "",
            "```json",
            JSON.stringify(delta, null, 2),
            "```",
            "",
        ].join("\n")
        const checksum = sha(content)
        const sequence = catalog.nextSequence
        const id = `c${String(sequence).padStart(6, "0")}-${checksum.slice(0, 12)}`
        const entry: ArchiveEntry = {
            id,
            sequence,
            relativePath: archiveRelativePath(input.sessionId, id),
            rangeHash: input.plan.rangeHash,
            firstMessageId: delta[0].info.id,
            lastMessageId: delta.at(-1)!.info.id,
            fingerprints: Object.fromEntries(
                delta.map((message) => [message.info.id, nativeMessageFingerprint(message)]),
            ),
            checksum,
            createdAt: new Date().toISOString(),
            status: "pending",
        }
        const full = authorizedArchivePath(input.directory, input.sessionId, entry)
        // If the previous attempt wrote the archive but crashed before publishing the
        // catalog, the ID is deterministic. Verify and reuse it instead of replacing it.
        let orphan: string | undefined
        try {
            orphan = await readPrivateFile(full, input.directory)
        } catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error
        }
        if (orphan !== undefined && sha(orphan) !== checksum)
            throw new Error("Archive ID collision or modified orphan")
        if (orphan === undefined) await writePrivateFile(full, content, input.directory)
        catalog.entries.push(entry)
        catalog.nextSequence = sequence + 1
        await persistCatalogUnderLock(input.directory, catalog)
        return { catalog, entry }
    })
}

export async function readArchiveEntry(
    directory: string,
    catalog: ArchiveCatalog,
    id: string,
): Promise<string> {
    const entry = catalog.entries.find((candidate) => candidate.id === id)
    if (!entry) throw new Error("unknown_archive")
    if (archiveExpired(entry)) throw new Error("expired_archive")
    const content = await readPrivateFile(
        authorizedArchivePath(directory, catalog.ownerSessionId, entry),
        directory,
    )
    if (sha(content) !== entry.checksum) throw new Error("missing_or_modified_file")
    return content
}

/** Preserve a valid-but-too-large result privately rather than discarding it. */
export async function archiveOversizedSummary(
    directory: string,
    catalog: ArchiveCatalog,
    archiveId: string,
    summary: string,
    associatedArchiveIds: string[] = catalog.entries
        // The returned handoff also carries the previous validated checkpoint.
        // Keep its described source IDs linked alongside this pending cohort.
        .filter((entry) => !archiveExpired(entry))
        .map((entry) => entry.id),
): Promise<void> {
    const latest = await loadArchiveCatalog(directory, catalog.ownerSessionId)
    const entry = latest.entries.find((candidate) => candidate.id === archiveId)
    if (!entry) throw new Error("unknown_archive")
    const relativePath = `${rootName}/sessions/${validSession(catalog.ownerSessionId)}/archives/${entry.id}.summary.md`
    const absolute = resolve(directory, relativePath)
    const base = relative(resolve(directory), absolute)
    if (base === ".." || base.startsWith(`..${sep}`) || isAbsolute(base))
        throw new Error("Archive path escapes project")
    await writePrivateFile(absolute, summary, directory)
    entry.oversizedSummaryPath = relativePath
    entry.oversizedSummaryChecksum = sha(summary)
    entry.oversizedSummaryArchiveIds = associatedArchiveIds.filter((id) =>
        latest.entries.some((candidate) => candidate.id === id),
    )
    await saveArchiveCatalog(directory, latest)
}

/** Rejected summaries are private evidence; validate their catalog path and bytes. */
export async function readArchivedSummary(
    directory: string,
    catalog: ArchiveCatalog,
    archiveId: string,
): Promise<{ path: string; text: string; checksum: string; archiveIds: string[] }> {
    const entry = catalog.entries.find((candidate) => candidate.id === archiveId)
    if (!entry) throw new Error("unknown_archive")
    if (archiveExpired(entry)) throw new Error("expired_archive")
    const path = authorizedArchivedSummaryPath(directory, catalog, archiveId)
    const text = await readPrivateFile(path, directory)
    if (sha(text) !== entry.oversizedSummaryChecksum) throw new Error("missing_or_modified_file")
    return {
        path,
        text,
        checksum: entry.oversizedSummaryChecksum,
        archiveIds: entry.oversizedSummaryArchiveIds ?? [archiveId],
    }
}

export function authorizedArchivedSummaryPath(
    directory: string,
    catalog: ArchiveCatalog,
    archiveId: string,
): string {
    const entry = catalog.entries.find((candidate) => candidate.id === archiveId)
    if (!entry) throw new Error("unknown_archive")
    const expected = `${rootName}/sessions/${validSession(catalog.ownerSessionId)}/archives/${entry.id}.summary.md`
    if (
        !entry.oversizedSummaryPath ||
        entry.oversizedSummaryPath !== expected ||
        !entry.oversizedSummaryChecksum
    )
        throw new Error("missing_or_modified_file")
    return resolve(directory, expected)
}
