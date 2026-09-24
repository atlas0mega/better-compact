import assert from "node:assert/strict"
import test from "node:test"
import { DEFAULT_CUSTOM_COMPACTION } from "@better-compact/core"
import {
    getInvalidConfigKeys,
    mergeCompaction,
    resolveModelConfig,
    validateConfigTypes,
    type PluginConfig,
} from "../lib/config"

function config(): PluginConfig {
    return {
        enabled: true,
        autoUpdate: false,
        debug: false,
        commands: { enabled: true },
        experimental: { allowSubAgents: false },
        compress: { permission: "allow" },
        compaction: {
            automatic: true,
            preset: "custom",
            summaryEffort: "inherit",
            custom: { ...DEFAULT_CUSTOM_COMPACTION },
            providers: {
                runpod: {
                    triggerTokens: 200000,
                    custom: { summarizerConcurrency: 3 },
                    models: {
                        "qwen/3.8": { triggerTokens: 181000 },
                        percent: { triggerTokens: null, custom: { triggerPercent: 70 } },
                    },
                },
            },
        },
    }
}

test("unspecified fields inherit model -> provider -> global independently", () => {
    const base = config(),
        before = structuredClone(base)
    const effective = resolveModelConfig(base, "runpod", "qwen/3.8")
    assert.equal(effective.compaction.triggerTokens, 181000)
    assert.equal(effective.compaction.custom.summarizerConcurrency, 3)
    assert.equal(effective.compaction.custom.targetPercent, base.compaction.custom.targetPercent)
    assert.equal(effective.compaction.summaryEffort, base.compaction.summaryEffort)
    assert.equal(effective.compaction.automatic, true)
    assert.deepEqual(base, before)
})

test("reasoning effort inherits independently at every scope", () => {
    const base = config()
    base.compaction.summaryEffort = "medium"
    base.compaction.providers!.runpod!.summaryEffort = "high"
    base.compaction.providers!.runpod!.models!["qwen/3.8"]!.summaryEffort = "low"
    assert.equal(resolveModelConfig(base, "runpod", "qwen/3.8").compaction.summaryEffort, "low")
    assert.equal(resolveModelConfig(base, "runpod", "other").compaction.summaryEffort, "high")
    assert.equal(resolveModelConfig(base, "other", "other").compaction.summaryEffort, "medium")
})

test("scratch summary model inherits and can be overridden or cleared independently", () => {
    const base = config()
    base.compaction.summaryModel = "openai/gpt-6-luna"
    base.compaction.providers!.runpod!.summaryModel = "google/gemini-2.5-flash"
    base.compaction.providers!.runpod!.models!["qwen/3.8"]!.summaryModel = "openai/gpt-6-sol"
    base.compaction.providers!.runpod!.models!.percent!.summaryModel = null
    assert.equal(
        resolveModelConfig(base, "runpod", "qwen/3.8").compaction.summaryModel,
        "openai/gpt-6-sol",
    )
    assert.equal(
        resolveModelConfig(base, "runpod", "other").compaction.summaryModel,
        "google/gemini-2.5-flash",
    )
    assert.equal(resolveModelConfig(base, "runpod", "percent").compaction.summaryModel, null)
    assert.equal(
        resolveModelConfig(base, "other", "other").compaction.summaryModel,
        "openai/gpt-6-luna",
    )
    assert.equal(base.compaction.summaryModel, "openai/gpt-6-luna")
})

test("unknown model inherits provider; unknown provider and missing identity inherit global", () => {
    const base = config()
    assert.equal(resolveModelConfig(base, "runpod", "other").compaction.triggerTokens, 200000)
    for (const id of [undefined, "other", "constructor", "__proto__"])
        assert.equal(resolveModelConfig(base, id, "qwen/3.8"), base)
    assert.equal(resolveModelConfig(base, "runpod", "constructor").compaction.triggerTokens, 200000)
})

test("model can clear absolute budgets without changing unrelated settings", () => {
    const effective = resolveModelConfig(config(), "runpod", "percent")
    assert.equal(effective.compaction.triggerTokens, null)
    assert.equal(effective.compaction.custom.triggerPercent, 70)
    assert.equal(effective.compaction.custom.summarizerConcurrency, 3)
})

test("config layers merge nested provider/model fields without losing global defaults", () => {
    const base = config()
    const merged = mergeCompaction(base.compaction, {
        providers: {
            runpod: {
                models: {
                    "qwen/3.8": { targetTokens: 90000, custom: { recentToolTokens: 12000 } },
                },
            },
        },
    })
    const effective = resolveModelConfig({ ...base, compaction: merged }, "runpod", "qwen/3.8")
    assert.equal(effective.compaction.triggerTokens, 181000)
    assert.equal(effective.compaction.targetTokens, 90000)
    assert.equal(effective.compaction.custom.summarizerConcurrency, 3)
    assert.equal(effective.compaction.custom.recentToolTokens, 12000)
    assert.equal(effective.compaction.custom.targetPercent, 35)
})

test("false and concurrent model resolution do not leak across sessions", async () => {
    const base = config()
    base.compaction.providers!.runpod!.models!["disabled"] = { automatic: false }
    const [disabled, qwen, global] = await Promise.all([
        Promise.resolve(resolveModelConfig(base, "runpod", "disabled")),
        Promise.resolve(resolveModelConfig(base, "runpod", "qwen/3.8")),
        Promise.resolve(resolveModelConfig(base, "other", "qwen/3.8")),
    ])
    assert.equal(disabled.compaction.automatic, false)
    assert.equal(qwen.compaction.automatic, true)
    assert.equal(global.compaction.triggerTokens, undefined)
    assert.equal(base.compaction.automatic, true)
})

test("validation supports exact IDs, partial overrides and null budgets", () => {
    const base = config()
    base.compaction.summaryModel = "openai/gpt-6-luna"
    base.compaction.providers!.runpod!.models!.percent!.summaryModel = null
    assert.deepEqual(getInvalidConfigKeys(base), [])
    assert.deepEqual(validateConfigTypes(base), [])
    assert.deepEqual(getInvalidConfigKeys(config()), [])
    assert.deepEqual(validateConfigTypes(config()), [])
})

test("validation rejects malformed maps and unsafe absolute budgets", () => {
    for (const providers of [
        null,
        [],
        true,
        { p: null },
        { p: { models: [] } },
        { p: { models: { m: false } } },
    ])
        assert.ok(validateConfigTypes({ compaction: { providers } }).length)
    for (const triggerTokens of [0, -1, 1.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1, "181000"])
        assert.ok(
            validateConfigTypes({
                compaction: { providers: { p: { models: { m: { triggerTokens } } } } },
            }).length,
        )
    assert.deepEqual(
        getInvalidConfigKeys({
            compaction: { providers: { "p.id": { models: { "m/id": { typo: true } } } } },
        }),
        ["compaction.providers.p.id.models.m/id.typo"],
    )
})

test("validation rejects malformed scratch summary model IDs at every scope", () => {
    for (const summaryModel of ["openai", "/model", "openai/", "openai/model name", 7, false, {}]) {
        assert.ok(
            validateConfigTypes({ compaction: { summaryModel } }).some(
                (error) => error.key === "compaction.summaryModel",
            ),
        )
        assert.ok(
            validateConfigTypes({
                compaction: { providers: { p: { models: { m: { summaryModel } } } } },
            }).some((error) => error.key === "compaction.providers.p.models.m.summaryModel"),
        )
    }
})
