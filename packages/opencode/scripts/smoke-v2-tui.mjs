import { spawn } from "node:child_process"
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const sandbox = await mkdtemp(path.join(tmpdir(), "better-compact-v2-tui-"))
const quoted = (value) => `'${value.replaceAll("'", "'\\''")}'`
let child
try {
    const project = path.join(sandbox, "project")
    const config = path.join(sandbox, "config", "opencode")
    const home = path.join(sandbox, "home")
    await Promise.all([mkdir(project), mkdir(config, { recursive: true }), mkdir(home)])
    await writeFile(path.join(config, "opencode.jsonc"), JSON.stringify({
        plugins: [root], compaction: { auto: true },
    }))
    const log = path.join(sandbox, "tui.typescript")
    const env = { ...process.env, HOME: home, XDG_CONFIG_HOME: path.dirname(config),
        XDG_DATA_HOME: path.join(sandbox, "data"), XDG_CACHE_HOME: path.join(sandbox, "cache"),
        OPENCODE_DB: path.join(sandbox, "staging.db"), OPENCODE_PASSWORD: "fixture-only",
        TERM: "xterm-256color" }
    const executable = process.env.OPENCODE_BIN ?? "opencode"
    child = spawn("script", ["-q", "-c", `TERM=xterm-256color ${quoted(executable)} --standalone --print-logs ${quoted(project)}`, log],
        { env, cwd: project, stdio: ["pipe", "pipe", "pipe"] })
    let output = ""
    let commandSent = false
    let commandDiscovered = false
    const onData = (chunk) => {
        output = (output + String(chunk)).slice(-150_000)
        if (!commandSent && output.includes("plugin reconciliation completed")) {
            commandSent = true
            setTimeout(() => child.stdin.write("/better-compact-status"), 800)
        }
        if (commandSent && !commandDiscovered && output.includes("Better Compact archive status")) {
            commandDiscovered = true
            setTimeout(() => child.stdin.write("\r"), 250)
            setTimeout(() => child.stdin.write("\r"), 800)
        }
    }
    child.stdout.on("data", onData)
    child.stderr.on("data", onData)
    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => resolve(), 10_000)
        child.once("error", reject)
        child.once("exit", (code) => { clearTimeout(timeout); reject(new Error(`V2 TUI exited prematurely: ${code}`)) })
    })
    const transcript = await readFile(log, "utf8")
    const seen = (text) => output.includes(text) || transcript.includes(text)
    if (!commandSent || !commandDiscovered || !seen("loading plugin") || !seen("better-compact") ||
        seen("Keymap.Provider is missing") || seen("Plugin failed:") ||
        seen("plugin operation failed"))
        throw new Error(`Real V2 TUI smoke failed: ${JSON.stringify({
            commandSent, commandDiscovered, loaded: seen("loading plugin") && seen("better-compact"),
            keymapError: seen("Keymap.Provider is missing"), pluginFailed: seen("Plugin failed:"),
            operationFailed: seen("plugin operation failed"), commandToast: seen("Open a session first"),
            sample: output.replace(/\x1b\[[\d;?]*[A-Za-z]/g, "").replace(/[\x00-\x1f]/g, " ").slice(-550),
        })}`)
    if (!seen("Open a session first"))
        throw new Error(`Better Compact's real TUI command did not run inside its app-slot keymap provider: ${JSON.stringify({
            typed: seen("better-compact-status"),
            sample: output.replace(/\x1b\[[\d;?]*[A-Za-z]/g, "").replace(/[\x00-\x1f]/g, " ").slice(-650),
        })}`)
    console.log("V2 TUI loaded Better Compact and ran /better-compact-status without a plugin failure.")
} finally {
    if (child && child.exitCode === null && child.signalCode === null) {
        const exited = new Promise((resolve) => child.once("exit", resolve))
        child.kill("SIGINT")
        await Promise.race([exited, new Promise((resolve) => setTimeout(() => { child.kill("SIGKILL"); resolve() }, 5_000))])
    }
    await rm(sandbox, { recursive: true, force: true })
}
