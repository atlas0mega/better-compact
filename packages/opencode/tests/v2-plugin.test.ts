import assert from "node:assert/strict"
import test from "node:test"
import { mkdir, mkdtemp, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import { join } from "node:path"
import { Message } from "@opencode/ai"
import type { Context } from "@opencode/plugin/promise/plugin"
import { loadV2Catalog, readV2Archive } from "../lib/v2/archive"
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

test("the V2 preflight capability requests fallback only when pruning cannot fit", async (t) => {
    const root = await mkdtemp(join(tmpdir(), "bc-preflight-capability-"))
    t.after(() => rm(root, { recursive: true, force: true }))
    const previous = process.env.XDG_CONFIG_HOME
    process.env.XDG_CONFIG_HOME = root
    t.after(() => {
        if (previous === undefined) delete process.env.XDG_CONFIG_HOME
        else process.env.XDG_CONFIG_HOME = previous
    })
    const configDir = join(root, "opencode")
    await mkdir(configDir)
    await writeFile(join(configDir, "better-compact.jsonc"), JSON.stringify({ compaction: {
        preset: "custom", summaryEffort: "off", custom: { triggerPercent: 1, targetPercent: 25,
            recentToolTokens: 0, prefixSummary: false },
    } }))
    const messages = [Message.user("Never overwrite the user files"),
        ...Array.from({ length: 12 }, (_, index) => [
            Message.make({ role: "assistant", content: [{ type: "tool-call", id: `old-${index}`, name: "read", input: {} }] }),
            Message.make({ role: "tool", content: [{ type: "tool-result", id: `old-${index}`, name: "read",
                result: { type: "text", value: "old data ".repeat(400) } }] }),
        ]).flat(), Message.user("The current requirement ".repeat(2_400))]
    for (const capability of [true, false]) {
        const hooks = new Map<string, (event: any) => Promise<void>>()
        const status = new Map<string, unknown>()
        const ctx = {
            location: { directory: root },
            tool: { transform: async (fn: (editor: any) => void) => fn({ add: () => {} }) },
            rpc: { register: async () => {} },
            session: {
                ...(capability ? { capabilities: { compactionFallback: true } } : {}),
                get: async () => ({ location: { directory: root } }),
                context: async () => [],
                hook: async (name: string, callback: (event: any) => Promise<void>) => { hooks.set(name, callback) },
            },
            model: { list: async () => ({ data: [{ providerID: "fixture", id: "fixture",
                limit: { context: 10000, input: 10000 }, settings: {}, variants: [] }] }) },
            storage: { set: async (key: string, value: unknown) => { status.set(key, value) } },
        } as unknown as Context
        await plugin.setup(ctx)
        const event: any = { sessionID: capability ? "ses_fallback" : "ses_older_host",
            model: { providerID: "fixture", id: "fixture" }, messages: [...messages], system: [] }
        await hooks.get("context")!(event)
        if (capability) {
            assert.deepEqual(event.compaction, { mode: "fallback", preserveTailMessages: 1 })
            assert.equal(JSON.stringify(event.messages), JSON.stringify(messages))
            assert.equal((status.get(`status/${event.sessionID}`) as { reason: string }).reason, "needs-native-fallback")
            const catalog = await loadV2Catalog(root, event.sessionID)
            const archived = (await Promise.all(catalog.entries.map((entry) =>
                readV2Archive(root, event.sessionID, entry.id)))).flat()
            assert.deepEqual(new Set(archived.map((message) => JSON.stringify(message))),
                new Set(messages.map((message) => JSON.stringify(message))))
            const replay: any = { sessionID: event.sessionID, model: event.model, system: [], messages: [
                Message.assistant([{ type: "compaction", provider: "openai", encrypted: "opaque-native-state" }]),
                Message.user("Another correction since that checkpoint"),
            ] }
            await hooks.get("context")!(replay)
            assert.match(replay.system[0]?.text ?? "", /better_compact_recall\(mode: "catalog"\)/)
        } else {
            assert.equal(event.compaction, undefined, "an older host cannot honor preflight fallback")
            assert.notEqual(JSON.stringify(event.messages), JSON.stringify(messages))
        }
    }
})
