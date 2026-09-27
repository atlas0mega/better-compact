import assert from "node:assert/strict"
import test from "node:test"
import tui from "../lib/v2/tui"

test("V2 TUI registers only native CLI commands and routes manual compaction through the V2 client", async () => {
    let layer: any
    const queued: string[] = []
    const notices: string[] = []
    const ctx = {
        client: {
            rpc: () => ({ status: async () => ({ archiveCount: 2, readyCount: 1, hasCheckpoint: true,
                retirementThrough: 0, lastStatus: "applied", beforeTokens: 1200, afterTokens: 500,
                triggerTokens: 800, targetTokens: 300 }) }),
            session: { compact: async ({ sessionID }: { sessionID: string }) => { queued.push(sessionID) } },
        },
        keymap: { layer: (create: () => unknown) => { layer = create() } },
        ui: {
            router: { current: () => ({ type: "session", sessionID: "ses_test" }) },
            toast: { show: (input: { message: string }) => { notices.push(input.message) } },
        },
    }
    await tui.setup(ctx as never)
    assert.equal(tui.id, "better-compact.tui")
    assert.deepEqual(layer.commands.map((item: any) => item.slash.name), ["better-compact", "better-compact-status"])
    await layer.commands[0].run()
    await layer.commands[1].run()
    assert.deepEqual(queued, ["ses_test"])
    assert.match(notices[0], /next safe point/)
    assert.match(notices[1], /2 exact archives/)
})
