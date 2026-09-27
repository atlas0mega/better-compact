import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp, mkdir, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { loadV2Config } from "../lib/v2/config"

test("V2 JSONC config preserves the user's custom budget and model overrides", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-config-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const global = join(root, "global")
    const project = join(root, "project")
    await mkdir(global)
    await mkdir(join(project, ".opencode"), { recursive: true })
    await writeFile(join(global, "better-compact.jsonc"), `{ "compaction": {
      "preset": "custom", "summaryModel": "openai/gpt-6-luna", "custom": {
        // tuned by the user
        "triggerPercent": 66, "targetPercent": 25, "recentReasoningTokens": 28000
      }, "providers": { "vast": { "models": { "qwen": { "triggerTokens": 144000 } } } }
    } }`)
    await writeFile(join(project, ".opencode", "better-compact.jsonc"), `{ "experimental": { "allowSubAgents": true } }`)
    const config = await loadV2Config(project, { providerID: "vast", id: "qwen" }, { global })
    assert.equal(config.compaction.summaryModel, "openai/gpt-6-luna")
    assert.equal(config.compaction.custom.triggerPercent, 66)
    assert.equal(config.compaction.custom.targetPercent, 25)
    assert.equal(config.compaction.custom.recentReasoningTokens, 28000)
    assert.equal(config.compaction.triggerTokens, 144000)
    assert.equal(config.experimental.allowSubAgents, true)
})

test("V2 config rejects malformed layers instead of silently using unsafe defaults", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-v2-config-"))
    t.after(async () => { const { rm } = await import("node:fs/promises"); await rm(root, { recursive: true, force: true }) })
    const global = join(root, "global")
    await mkdir(global)
    await writeFile(join(global, "better-compact.jsonc"), `{ "compaction": { "targetTokens": -1 } }`)
    await assert.rejects(loadV2Config(root, undefined, { global }), /Invalid Better Compact V2 settings/)
})
