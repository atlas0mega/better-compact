import { spawn } from "node:child_process"
import { createServer } from "node:http"
import { createServer as createTcpServer } from "node:net"
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

const plugin = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const executable = process.env.OPENCODE_BIN
if (!executable) throw new Error("Set OPENCODE_BIN to a V2 host with the preflight compaction capability")
const sandbox = await mkdtemp(path.join(tmpdir(), "better-compact-native-fallback-"))
const primary = []
const checkpoints = []
const oldAssistant = `Completed historical work. ${"Old implementation detail ".repeat(3_000)}`
let host
const provider = createServer(async (request, response) => {
    if (request.url === "/v1/models") {
        response.writeHead(200, { "content-type": "application/json" })
        response.end(JSON.stringify({ object: "list", data: [{ id: "fixture", object: "model", created: 0, owned_by: "openai" }] }))
        return
    }
    if (request.url !== "/v1/responses" && request.url !== "/v1/responses/compact") {
        response.writeHead(404).end(); return
    }
    if (request.method !== "POST") { response.writeHead(200).end("ok"); return }
    const chunks = []
    for await (const chunk of request) chunks.push(chunk)
    if (!chunks.length || !Buffer.concat(chunks).length) { response.writeHead(400).end("empty body"); return }
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"))
    const compact = request.url === "/v1/responses/compact"
    if (compact) {
        checkpoints.push(body)
        if (JSON.stringify(body.input).includes("REJECT_PREFIX_FIXTURE")) {
            response.writeHead(400, { "content-type": "application/json" })
            response.end(JSON.stringify({ error: { type: "invalid_request_error", code: "context_length_exceeded",
                message: "Selected native compaction prefix is too long" } }))
            return
        }
        response.writeHead(200, { "content-type": "application/json" })
        response.end(JSON.stringify({ object: "response.compaction", output: [
            { type: "compaction", id: "cmp_fixture", encrypted_content: "encrypted-native-fixture" },
        ], usage: { input_tokens: 500, output_tokens: 100, total_tokens: 600 } }))
        return
    }
    primary.push(body)
    response.writeHead(200, { "content-type": "text/event-stream" })
    if (!compact) {
        const id = `msg_fixture_${primary.length}`
        response.write(`data: ${JSON.stringify({ type: "response.output_item.added", item: { type: "message", id } })}\n\n`)
        response.write(`data: ${JSON.stringify({ type: "response.output_text.delta", item_id: id,
            delta: primary.length === 1 || JSON.stringify(body.input).includes("REJECT_PREFIX_FIXTURE")
                ? oldAssistant : "The current task continues safely." })}\n\n`)
    }
    response.write(`data: ${JSON.stringify({ type: "response.completed", response: {
        id: `resp_fixture_${primary.length}_${checkpoints.length}`, status: "completed",
        output: [],
        usage: { input_tokens: 500, output_tokens: 100, total_tokens: 600 },
    } })}\n\n`)
    response.end("data: [DONE]\n\n")
})

async function run(project, env, args, maxOutput = 12_000) {
    const child = spawn(executable, args, { cwd: project, env, stdio: ["ignore", "pipe", "pipe"] })
    let output = ""
    let errors = ""
    child.stdout.on("data", (chunk) => { output = (output + String(chunk)).slice(-maxOutput) })
    child.stderr.on("data", (chunk) => { errors = (errors + String(chunk)).slice(-12_000) })
    const exit = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => { child.kill(); reject(new Error("Native fallback host timed out")) }, 60_000)
        child.once("error", reject)
        child.once("exit", (code) => { clearTimeout(timeout); resolve(code) })
    })
    if (exit !== 0) throw new Error(`Native fallback host exited ${exit}: ${errors.slice(-2_000)}`)
    return output
}

try {
    const port = await new Promise((resolve) => provider.listen(0, "127.0.0.1", () => resolve(provider.address().port)))
    const project = path.join(sandbox, "project")
    const configHome = path.join(sandbox, "config")
    await mkdir(path.join(configHome, "opencode"), { recursive: true })
    await mkdir(path.join(project, ".opencode"), { recursive: true })
    await mkdir(path.join(sandbox, "home"))
    await writeFile(path.join(configHome, "opencode", "better-compact.jsonc"), JSON.stringify({
        compaction: { preset: "custom", summaryEffort: "off", custom: {
            triggerPercent: 1, targetPercent: 25, recentToolTokens: 0, prefixSummary: false,
        } },
    }))
    await writeFile(path.join(project, "opencode.jsonc"), JSON.stringify({
        $schema: "https://opencode.ai/config.json", model: "openai/fixture", plugins: [plugin],
        compaction: { auto: false, keep: { tokens: 3_000 } },
        providers: { openai: { settings: { apiKey: "fixture-only", baseURL: `http://127.0.0.1:${port}/v1`,
            compaction: { type: "native" } }, models: { fixture: { name: "Fixture", capabilities: {
                tools: true, input: ["text"], output: ["text"],
            }, limit: { context: 12_000, output: 2_048 } } } } },
    }))
    const env = { ...process.env, HOME: path.join(sandbox, "home"), XDG_CONFIG_HOME: configHome,
        XDG_DATA_HOME: path.join(sandbox, "data"), XDG_CACHE_HOME: path.join(sandbox, "cache"),
        OPENCODE_DB: path.join(sandbox, "fixture.db"), OPENCODE_PASSWORD: "fixture-only" }
    const apiPort = await new Promise((resolve) => {
        const server = createTcpServer()
        server.listen(0, "127.0.0.1", () => { const value = server.address().port; server.close(() => resolve(value)) })
    })
    const startHost = async () => {
        host = spawn(executable, ["serve", "--hostname", "127.0.0.1", "--port", String(apiPort)],
            { cwd: project, env, stdio: ["ignore", "pipe", "pipe"] })
        await new Promise((resolve, reject) => {
            const timeout = setTimeout(() => reject(new Error("Private native-fallback server did not start")), 30_000)
            host.once("error", reject)
            host.once("exit", (code) => reject(new Error(`Private native-fallback server exited ${code}`)))
            host.stdout.on("data", (chunk) => {
                if (!String(chunk).includes("server listening")) return
                clearTimeout(timeout)
                resolve()
            })
        })
    }
    await startHost()
    const address = `http://127.0.0.1:${apiPort}`
    const sessionID = JSON.parse(await run(project, env, ["api", "--server", address, "post", "/api/session", "--data",
        '{"title":"Native fallback smoke"}'])).data?.id
    if (!sessionID) throw new Error("No private fallback session was created")
    await run(project, env, ["run", "--server", address, "--session", sessionID, "--model", "openai/fixture",
        "Never overwrite user files; keep the original task trajectory."])
    if (primary.length !== 1) throw new Error(`Expected one initial provider request, saw ${primary.length}`)
    await run(project, env, ["run", "--server", address, "--session", sessionID, "--model", "openai/fixture",
        "Latest human correction: finish the active test without dropping the prior constraint."])
    const stored = JSON.parse(await run(project, env, ["api", "--server", address,
        "post", "/api/rpc/better-compact/status", "--data", JSON.stringify({ input: { sessionID } })])).output
    if (primary.length !== 2 || checkpoints.length !== 1)
        throw new Error(`Native fallback order was not one primary, one checkpoint, one resumed primary: ${JSON.stringify({
            primary: primary.length, checkpoints: checkpoints.length, status: stored,
            primaryLengths: primary.map((item) => JSON.stringify(item.input).length),
        })}`)
    const compacted = JSON.stringify(checkpoints[0].input)
    const continued = JSON.stringify(primary[1].input)
    if (compacted.includes("compaction_trigger") || !compacted.includes(oldAssistant) ||
        !continued.includes("encrypted-native-fixture") || continued.includes("Old implementation detail") ||
        !continued.includes("Never overwrite user files") || !continued.includes("Latest human correction"))
        throw new Error(`Provider-visible native fallback lost human intent or replayed the old assistant payload: ${JSON.stringify({
            compactedEndpoint: !compacted.includes("compaction_trigger"),
            compactedHistoricalPrefix: compacted.includes(oldAssistant),
            continuedEncrypted: continued.includes("encrypted-native-fixture"),
            continuedOld: continued.includes("Old implementation detail"),
            originalUser: continued.includes("Never overwrite user files"),
            latestUser: continued.includes("Latest human correction"), status: stored,
        })}`)
    const durable = JSON.parse(await run(project, env, ["api", "--server", address,
        "get", `/api/session/${sessionID}/message`], 200_000)).data
    if (!durable?.some((message) => JSON.stringify(message).includes(oldAssistant)))
        throw new Error("Native compaction did not retain the exact original assistant message in durable history")
    const archiveRoot = path.join(project, ".opencode", "better-compact", "v2", "sessions", sessionID)
    const catalog = JSON.parse(await readFile(path.join(archiveRoot, "catalog.json"), "utf8"))
    const archived = await Promise.all(catalog.entries.filter((entry) => entry.status !== "expired")
        .map((entry) => readFile(path.join(archiveRoot, "archives", `${entry.id}.json`), "utf8")))
    if (!archived.some((text) => text.includes(oldAssistant)) ||
        !JSON.stringify(primary[1]).includes("better_compact_recall"))
        throw new Error("Native fallback did not preserve an exact, provider-discoverable recall path")
    if (stored?.lastStatus === "error") throw new Error(`Better Compact failed in private host: ${stored.reason}`)
    const log = await readFile(path.join(sandbox, "data", "opencode", "log", "opencode.log"), "utf8").catch(() => "")
    if (!log.includes("better-compact")) throw new Error("Private V2 host did not activate Better Compact")
    const stopped = new Promise((resolve) => host.once("exit", resolve))
    host.kill()
    await stopped
    await startHost()
    await run(project, env, ["run", "--server", address, "--session", sessionID, "--model", "openai/fixture",
        "Resume after restart; retain the original constraint and encrypted checkpoint."])
    const restarted = JSON.stringify(primary.at(-1)?.input)
    if (checkpoints.length !== 1 || !restarted.includes("encrypted-native-fixture") ||
        !restarted.includes("Never overwrite user files") || !restarted.includes("Resume after restart") ||
        restarted.includes("Old implementation detail"))
        throw new Error("Native fallback replay or human constraints were lost across a server restart")
    const rejectedID = JSON.parse(await run(project, env, ["api", "--server", address, "post", "/api/session", "--data",
        '{"title":"Reject selective native range"}'])).data?.id
    if (!rejectedID) throw new Error("No rejection fixture session was created")
    await run(project, env, ["run", "--server", address, "--session", rejectedID, "--model", "openai/fixture",
        "REJECT_PREFIX_FIXTURE Keep this human constraint visible."])
    const primariesBefore = primary.length
    const checkpointsBefore = checkpoints.length
    await run(project, env, ["run", "--server", address, "--session", rejectedID, "--model", "openai/fixture",
        "Do not send an uncompactable primary or drop older evidence."]).catch(() => undefined)
    const rejectedHistory = JSON.parse(await run(project, env, ["api", "--server", address,
        "get", `/api/session/${rejectedID}/message`], 200_000)).data
    if (primary.length !== primariesBefore || checkpoints.length !== checkpointsBefore + 1 ||
        rejectedHistory.some((message) => message.info?.type === "compaction" && message.info?.status === "completed"))
        throw new Error("An invalid native range was retried as a flattened checkpoint or raw primary request")
    console.log("Private host: native prefix checkpoint, encrypted replay across restart, human constraints and exact recoverable archive verified.")
} finally {
    if (host && host.exitCode === null && host.signalCode === null) {
        const stopped = new Promise((resolve) => host.once("exit", resolve))
        host.kill()
        await stopped
    }
    await new Promise((resolve) => provider.close(resolve))
    await rm(sandbox, { recursive: true, force: true })
}
