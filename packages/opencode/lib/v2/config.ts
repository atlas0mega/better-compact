import * as fs from "node:fs/promises"
import { homedir } from "node:os"
import { dirname, join, resolve } from "node:path"
import { parse, type ParseError } from "jsonc-parser"
import { DEFAULT_CUSTOM_COMPACTION, normalizeCompactionCustom, normalizePreset, normalizeSummaryEffort,
    type CompactionConfig, type CompactionCustomSettings } from "@better-compact/core"

type CompactionOverride = Partial<Omit<CompactionConfig, "custom">> & {
    custom?: Partial<CompactionCustomSettings>
    summaryModel?: string | null
    triggerTokens?: number | null
    targetTokens?: number | null
}
type ProviderOverride = CompactionOverride & { models?: Record<string, CompactionOverride> }
export type ScopedCompaction = CompactionConfig & {
    summaryModel?: string | null
    triggerTokens?: number | null
    targetTokens?: number | null
    providers?: Record<string, ProviderOverride>
}
export interface PluginConfig {
    enabled: boolean
    debug: boolean
    commands: { enabled: boolean }
    experimental: { allowSubAgents: boolean }
    compress: { permission: "allow" | "ask" | "deny" }
    compaction: ScopedCompaction
}

function defaults(): PluginConfig {
    return { enabled: true, debug: false,
        commands: { enabled: true }, experimental: { allowSubAgents: false },
        compress: { permission: "allow" },
        compaction: { automatic: true, preset: "light", summaryEffort: "inherit",
            custom: { ...DEFAULT_CUSTOM_COMPACTION } },
    }
}

const scalar = new Set(["automatic", "preset", "summaryEffort", "summaryModel", "triggerTokens", "targetTokens", "custom"])
const custom = new Set(["triggerPercent", "targetPercent", "recentToolTokens", "recentReasoningTokens",
    "summarizerConcurrency", "prefixSummary", "collapsePercent"])
const record = (value: unknown): value is Record<string, unknown> =>
    !!value && typeof value === "object" && !Array.isArray(value)
const keys = (value: Record<string, unknown>, allowed: Set<string>) =>
    Object.keys(value).every((key) => allowed.has(key))
const optional = (value: unknown, type: "boolean" | "string") => value === undefined || typeof value === type
const positive = (value: unknown) => value === undefined || value === null ||
    typeof value === "number" && Number.isSafeInteger(value) && value > 0

function validOverride(value: unknown, allowModels = false, allowProviders = false): boolean {
    if (!record(value) || !keys(value, new Set([...scalar, ...(allowModels ? ["models"] : []),
        ...(allowProviders ? ["providers"] : [])]))) return false
    if (!optional(value.automatic, "boolean") || !positive(value.triggerTokens) || !positive(value.targetTokens)) return false
    if (value.preset !== undefined && !["light", "moderate", "max", "custom"].includes(String(value.preset))) return false
    if (value.summaryEffort !== undefined && !["inherit", "low", "medium", "high", "max", "off"].includes(String(value.summaryEffort))) return false
    if (value.summaryModel !== undefined && value.summaryModel !== null &&
        (typeof value.summaryModel !== "string" || !/^[a-zA-Z0-9_.-]+\/[a-zA-Z0-9_.-]+$/.test(value.summaryModel))) return false
    if (value.custom !== undefined && (!record(value.custom) || !keys(value.custom, custom) ||
        Object.entries(value.custom).some(([name, item]) => name === "prefixSummary"
            ? typeof item !== "boolean" : typeof item !== "number" || !Number.isFinite(item)))) return false
    if (value.models !== undefined && (!record(value.models) || !Object.values(value.models).every((item) => validOverride(item)))) return false
    return true
}

function validateLayer(layer: unknown): asserts layer is Record<string, unknown> {
    if (!record(layer) || !keys(layer, new Set(["$schema", "enabled", "debug", "commands", "experimental", "compress", "compaction"])) ||
        !optional(layer.enabled, "boolean") || !optional(layer.debug, "boolean"))
        throw new Error("Invalid Better Compact V2 settings")
    for (const [name, field] of [["commands", "enabled"], ["experimental", "allowSubAgents"]] as const) {
        const value = layer[name]
        if (value !== undefined && (!record(value) || !keys(value, new Set([field])) || !optional(value[field], "boolean")))
            throw new Error("Invalid Better Compact V2 settings")
    }
    const permission = layer.compress
    if (permission !== undefined && (!record(permission) || !keys(permission, new Set(["permission"])) ||
        permission.permission !== undefined && !["allow", "ask", "deny"].includes(String(permission.permission))))
        throw new Error("Invalid Better Compact V2 settings")
    const compaction = layer.compaction
    if (compaction !== undefined && (!validOverride(compaction, false, true) || !record(compaction) ||
        compaction.providers !== undefined && (!record(compaction.providers) ||
            !Object.values(compaction.providers).every((value) => validOverride(value, true)))))
        throw new Error("Invalid Better Compact V2 settings")
}

function mergeOverride(a: CompactionOverride = {}, b: CompactionOverride = {}): CompactionOverride {
    return { ...a, ...b, ...(a.custom || b.custom ? { custom: { ...a.custom, ...b.custom } } : {}) }
}

function mergeProviders(a: ScopedCompaction["providers"] = {}, b: ScopedCompaction["providers"] = {}) {
    return Object.fromEntries([...new Set([...Object.keys(a), ...Object.keys(b)])].map((id) => {
        const left = a[id] ?? {}
        const right = b[id] ?? {}
        const models = Object.fromEntries([...new Set([...Object.keys(left.models ?? {}), ...Object.keys(right.models ?? {})])]
            .map((key) => [key, mergeOverride(left.models?.[key], right.models?.[key])]))
        return [id, { ...mergeOverride(left, right), models }]
    })) as Record<string, ProviderOverride>
}

function mergeCompaction(base: ScopedCompaction, override: CompactionOverride & { providers?: ScopedCompaction["providers"] } = {}): ScopedCompaction {
    return { ...base, ...override,
        preset: normalizePreset(override.preset ?? base.preset),
        summaryEffort: normalizeSummaryEffort(override.summaryEffort ?? base.summaryEffort),
        custom: normalizeCompactionCustom({ ...base.custom, ...override.custom }),
        ...(base.providers || override.providers ? { providers: mergeProviders(base.providers, override.providers) } : {}),
    }
}

function resolveModelConfig(config: PluginConfig, model: { providerID: string; id: string }): PluginConfig {
    const provider = config.compaction.providers?.[model.providerID]
    if (!provider) return config
    const { models, ...settings } = provider
    return { ...config, compaction: mergeCompaction(mergeCompaction(config.compaction, settings), models?.[model.id]) }
}

async function firstFile(directory: string): Promise<string | null> {
    for (const name of ["better-compact.jsonc", "better-compact.json"]) {
        const path = join(directory, name)
        try { await fs.access(path); return path }
        catch (error) { if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error }
    }
    return null
}

async function projectFile(directory: string): Promise<string | null> {
    let current = resolve(directory)
    for (;;) {
        const base = join(current, ".opencode")
        try {
            if ((await fs.stat(base)).isDirectory()) return firstFile(base)
        } catch (error) {
            if (!(error instanceof Error && "code" in error && error.code === "ENOENT")) throw error
        }
        const parent = dirname(current)
        if (parent === current) return null
        current = parent
    }
}

/** Reads the existing Better Compact JSONC directly for the V2 plugin. */
export async function loadV2Config(directory: string, model?: { providerID: string; id: string },
    dirs: { global?: string; config?: string } = {}): Promise<PluginConfig> {
    const globalDir = dirs.global ?? join(process.env.XDG_CONFIG_HOME ?? join(homedir(), ".config"), "opencode")
    const paths = [await firstFile(globalDir),
        ...(dirs.config ?? process.env.OPENCODE_CONFIG_DIR ? [await firstFile(dirs.config ?? process.env.OPENCODE_CONFIG_DIR!)] : []),
        await projectFile(directory)]
    let config = defaults()
    for (const path of paths) {
        if (!path) continue
        const errors: ParseError[] = []
        const data: unknown = parse(await fs.readFile(path, "utf8"), errors, { allowTrailingComma: true })
        if (errors.length) throw new Error("Invalid Better Compact V2 configuration")
        validateLayer(data)
        const layer = data
        const commands = layer.commands as Partial<PluginConfig["commands"]> | undefined
        const experimental = layer.experimental as Partial<PluginConfig["experimental"]> | undefined
        const compress = layer.compress as Partial<PluginConfig["compress"]> | undefined
        config = {
            ...config,
            enabled: (layer.enabled as boolean | undefined) ?? config.enabled,
            debug: (layer.debug as boolean | undefined) ?? config.debug,
            commands: { enabled: commands?.enabled ?? config.commands.enabled },
            experimental: { allowSubAgents: experimental?.allowSubAgents ?? config.experimental.allowSubAgents },
            compress: { permission: compress?.permission ?? config.compress.permission },
            compaction: mergeCompaction(config.compaction, layer.compaction as ScopedCompaction | undefined),
        }
    }
    return model ? resolveModelConfig(config, model) : config
}
