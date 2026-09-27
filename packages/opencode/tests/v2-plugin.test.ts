import assert from "node:assert/strict"
import test from "node:test"
import { mkdtemp, rm } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import type { Context } from "@opencode/plugin/promise/plugin"
import plugin from "../lib/v2/plugin"

test("V2 compaction hook defers opaque provider state to the host instead of failing the session", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-opaque-compaction-"))
    t.after(() => rm(root, { recursive: true, force: true }))
    const previousConfig = process.env.XDG_CONFIG_HOME
    const previousOverride = process.env.OPENCODE_CONFIG_DIR
    process.env.XDG_CONFIG_HOME = root
    delete process.env.OPENCODE_CONFIG_DIR
    t.after(() => {
        if (previousConfig === undefined) delete process.env.XDG_CONFIG_HOME
        else process.env.XDG_CONFIG_HOME = previousConfig
        if (previousOverride === undefined) delete process.env.OPENCODE_CONFIG_DIR
        else process.env.OPENCODE_CONFIG_DIR = previousOverride
    })

    let compaction: ((event: any) => Promise<void>) | undefined
    let providerNativeCompaction = false
    const ctx = {
        location: { directory: root },
        tool: { transform: async (fn: (editor: any) => void) => fn({ add: () => {} }) },
        rpc: { register: async () => {} },
        session: {
            get: async () => ({ location: { directory: root } }),
            hook: async (kind: string, fn: (event: any) => Promise<void>) => {
                if (kind === "compaction") compaction = fn
            },
        },
        model: { list: async () => ({ data: [{ providerID: "fixture", id: "fixture", limit: { context: 100000 }, settings: {}, variants: [] }] }) },
        provider: { get: async () => ({ data: { settings: providerNativeCompaction ? { compaction: { type: "native" } } : {} } }) },
    } as unknown as Context
    await plugin.setup(ctx)
    assert.ok(compaction)

    for (const opaque of [
        { type: "reasoning", text: "", encrypted: "provider-reasoning" },
        { type: "compaction", provider: "openai", encrypted: "provider-checkpoint" },
    ]) {
        const event: any = {
            sessionID: "ses_opaque", model: { providerID: "fixture", id: "fixture" },
            messages: [Message.make({ id: "u1", role: "user", content: [Message.text("Keep the goal intact")] }),
                Message.make({ id: "a1", role: "assistant", content: [opaque as never] })],
        }
        await assert.doesNotReject(() => compaction!(event))
        assert.equal(event.result, undefined, "host must own the fallback compaction")
    }

    providerNativeCompaction = true
    const nativeEvent: any = {
        sessionID: "ses_native", model: { providerID: "fixture", id: "fixture" },
        messages: [Message.make({ id: "u2", role: "user", content: [Message.text("Use provider-native compaction")] })],
    }
    await assert.doesNotReject(() => compaction!(nativeEvent))
    assert.equal(nativeEvent.result, undefined, "provider-level native setting must bypass the plugin handoff")
})
