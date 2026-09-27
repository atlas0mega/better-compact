import { execFileSync, spawn } from "node:child_process"
import { createServer } from "node:net"
import { mkdir, mkdtemp, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath, pathToFileURL } from "node:url"

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const executable = process.env.OPENCODE_BIN ?? "opencode"
const packageSpec = process.env.BETTER_COMPACT_SMOKE_SPEC
const sandbox = await mkdtemp(path.join(tmpdir(), "better-compact-v2-install-"))
let server
let diagnostics = ""

try {
    let target = packageSpec
    if (!target) {
        const metadata = JSON.parse(execFileSync("npm", ["pack", "--pack-destination", sandbox, "--json"], {
            cwd: root, encoding: "utf8",
        }))
        const packageInfo = Array.isArray(metadata) ? metadata[0] : Object.values(metadata)[0]
        if (!packageInfo?.filename) throw new Error("npm pack did not produce a package")
        target = path.join(sandbox, packageInfo.filename)
    }
    // The OpenCode TUI supplies these optional host peers at runtime. Install
    // them in the isolated Node smoke to import the packed TUI entry directly.
    execFileSync("npm", ["install", "--prefix", sandbox, "--no-save", "--ignore-scripts", "--no-audit", "--no-fund",
        target, "solid-js@1.9.12", "@opentui/core@0.5.12", "@opentui/solid@0.5.12",
        "@opencode/theme@2.0.18"], { cwd: sandbox, encoding: "utf8", timeout: 90_000 })
    const installed = path.join(sandbox, "node_modules", "better-compact")
    const packagedTui = await import(pathToFileURL(path.join(installed, "dist", "tui.js")).href)
    const packagedRPC = await import(pathToFileURL(path.join(installed, "dist", "rpc.js")).href)
    if (packagedTui.default?.id !== "better-compact.tui" || typeof packagedTui.default.setup !== "function" ||
        packagedRPC.BetterCompactRPC?.id !== "better-compact")
        throw new Error("Installed package TUI or RPC export is not V2-ready")
    const configHome = path.join(sandbox, "config")
    const configDir = path.join(configHome, "opencode")
    const projectDir = path.join(sandbox, "project")
    await Promise.all([mkdir(configDir, { recursive: true }), mkdir(projectDir), mkdir(path.join(sandbox, "home"))])
    await mkdir(path.join(projectDir, ".opencode"))
    await writeFile(path.join(projectDir, "opencode.jsonc"), `{
  "$schema": "https://opencode.ai/config.json",
  "plugins": [${JSON.stringify(installed)}]
}\n`)
    const marker = "// preserve this V2 JSONC comment"
    await writeFile(path.join(configDir, "opencode.jsonc"), `{
  ${marker}
  "$schema": "https://opencode.ai/config.json",
  "compaction": { "auto": false }
}\n`)
    const env = { ...process.env,
        HOME: path.join(sandbox, "home"), XDG_CONFIG_HOME: configHome,
        XDG_DATA_HOME: path.join(sandbox, "data"), XDG_CACHE_HOME: path.join(sandbox, "cache"),
        XDG_STATE_HOME: path.join(sandbox, "state"), OPENCODE_DB: path.join(sandbox, "staging.db"),
        OPENCODE_PASSWORD: "better-compact-staging-only",
    }
    const port = await new Promise((resolve, reject) => {
        const listener = createServer()
        listener.once("error", reject)
        listener.listen(0, "127.0.0.1", () => {
            const number = listener.address().port
            listener.close(() => resolve(number))
        })
    })
    server = spawn(executable, ["serve", "--hostname", "127.0.0.1", "--port", String(port), "--log-level", "debug", "--print-logs"],
        { cwd: projectDir, env, stdio: ["ignore", "pipe", "pipe"] })
    server.stderr.on("data", (chunk) => { diagnostics = (diagnostics + String(chunk)).slice(-12_000) })
    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("V2 server startup timed out")), 30_000)
        server.once("error", reject)
        server.once("exit", (code) => reject(new Error(`V2 server exited before startup: ${code}`)))
        server.stdout.on("data", (chunk) => {
            if (!String(chunk).includes("server listening")) return
            clearTimeout(timeout)
            resolve()
        })
    })
    const address = `http://127.0.0.1:${port}`
    execFileSync(executable, ["api", "--server", address, "post", "/api/session", "--data",
        '{"title":"Better Compact packaged smoke"}'], { cwd: projectDir, env, encoding: "utf8", timeout: 30_000 })
    const configDocs = execFileSync(executable, ["api", "--server", address, "get", "/api/config"],
        { cwd: projectDir, env, encoding: "utf8", timeout: 30_000 })
    let listing
    let plugin
    for (let attempt = 0; attempt < 12; attempt++) {
        listing = JSON.parse(execFileSync(executable, ["api", "--server", address, "get", "/api/plugin",
            "--param", `location.directory=${projectDir}`], { cwd: projectDir, env, encoding: "utf8", timeout: 60_000 }))
        plugin = listing.data?.find((item) => item.id === "better-compact")
        if (plugin?.state?.status === "active" || plugin?.state?.status === "error") break
        await new Promise((resolve) => setTimeout(resolve, 250))
    }
    if (plugin?.state?.status !== "active" || plugin.source?.type !== "local") {
        const serverLog = await readFile(path.join(sandbox, "data", "opencode", "log", "opencode.log"), "utf8").catch(() => "")
        const relevant = serverLog.split("\n").filter((line) => /plugin|error|warn/i.test(line)).slice(-30).join("\n")
        throw new Error(`Packed V2 plugin not active: ${plugin?.state?.status ?? "absent"}\nConfig: ${configDocs}\n${diagnostics}\n${relevant}`)
    }
    if (!(await readFile(path.join(configDir, "opencode.jsonc"), "utf8")).includes(marker))
        throw new Error("V2 config lost its JSONC comment")
    const installedManifest = JSON.parse(await readFile(path.join(installed, "package.json"), "utf8"))
    console.log(`Installed better-compact V2 ${installedManifest.version} loaded as an active server plugin from ${plugin.source.path}`)
} finally {
    if (server && server.exitCode === null && server.signalCode === null) {
        const stopped = new Promise((resolve) => server.once("exit", resolve))
        server.kill()
        await stopped
    }
    await rm(sandbox, { recursive: true, force: true })
}
