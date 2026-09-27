import { spawn } from "node:child_process"
import { createHash } from "node:crypto"
import { createServer } from "node:http"
import { createServer as createTcpServer } from "node:net"
import { mkdtemp, mkdir, readFile, rm, writeFile } from "node:fs/promises"
import { tmpdir } from "node:os"
import path from "node:path"
import { fileURLToPath } from "node:url"

const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))
const sandbox = await mkdtemp(path.join(tmpdir(), "better-compact-v2-host-"))
const requests = []
const summaries = []
const probes = []
let host
const server = createServer(async (req, res) => {
    probes.push(req.url)
    const chunks = []
    for await (const chunk of req) chunks.push(chunk)
    if (req.url === "/health") { res.writeHead(200).end("ok"); return }
    if (req.url === "/v1/models") {
        res.writeHead(200, { "content-type": "application/json" })
        res.end(JSON.stringify({ object: "list", data: [{ id: "mock", object: "model", created: 0, owned_by: "fixture" }] }))
        return
    }
    if (!req.url?.endsWith("/chat/completions")) {
        res.writeHead(404).end()
        return
    }
    const body = JSON.parse(Buffer.concat(chunks).toString("utf8"))
    const textInput = JSON.stringify(body.messages ?? [])
    const isHandoff = textInput.includes("Extract a chronology-aware task-state handoff")
    const isDescription = textInput.includes("Describe ONLY the historical delta archive")
    if (isHandoff || isDescription) summaries.push({ kind: isHandoff ? "handoff" : "description", request: body })
    else requests.push(body)
    const first = !isHandoff && !isDescription && [1, 4].includes(requests.length)
    const handoff = ["## Decisions", "- Keep the current test decision while checking the old diagnostic.",
        "## Files & Symbols", "- diagnostic.txt contains the archived tool evidence.",
        "## Errors (verbatim)", "- No active failure has been verified.",
        "## What failed and why", "- Repeating old read output would hide the implementation state.",
        "## Constraints", "- Never overwrite user files; keep the current test decision.",
        "- Continue the implementation without dropping the original constraint.",
        "## Next step", "- Verify the implementation and the original human constraints."].join("\n")
    const text = isHandoff ? JSON.stringify({ handoff }) : isDescription
        ? "Historical implementation decision, original constraints and the archived diagnostic.txt tool result"
        : requests.length >= 7
            ? `Implementation decision ${Math.max(1, requests.length - 6)}: diagnostic.txt was inspected; tests remain next. ${"Historical routine observation. ".repeat(350)}`
            : "The original human constraint remains in context and the next step is clear."
    res.writeHead(200, { "content-type": "text/event-stream" })
    const id = `mock-${requests.length}-${summaries.length}`
    res.write(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: 0, model: "fixture",
        choices: [{ index: 0, delta: first
            ? { role: "assistant", tool_calls: [{ index: 0, id: "fixture-read-1", type: "function",
                function: { name: "read", arguments: JSON.stringify({ path: "./diagnostic.txt" }) } }] }
            : { role: "assistant", content: text }, finish_reason: null }] })}\n\n`)
    res.write(`data: ${JSON.stringify({ id, object: "chat.completion.chunk", created: 0, model: "fixture",
        choices: [{ index: 0, delta: {}, finish_reason: first ? "tool_calls" : "stop" }],
        usage: { prompt_tokens: first ? 700 : 1100, completion_tokens: 200, total_tokens: 900 } })}\n\n`)
    res.end("data: [DONE]\n\n")
})

async function run(executable, env, project, args) {
    const child = spawn(executable, args, { cwd: project, env, stdio: ["ignore", "pipe", "pipe"] })
    let output = ""
    let errors = ""
    child.stdout.on("data", (chunk) => { output = (output + String(chunk)).slice(-10_000) })
    child.stderr.on("data", (chunk) => { errors = (errors + String(chunk)).slice(-10_000) })
    const code = await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => { child.kill(); reject(new Error("V2 mock host timed out")) }, 60_000)
        child.once("error", reject)
        child.once("exit", (status) => { clearTimeout(timeout); resolve(status) })
    })
    if (code !== 0) {
        const log = await readFile(path.join(path.dirname(env.OPENCODE_DB), "data", "opencode", "log", "opencode.log"), "utf8").catch(() => "")
        const relevant = log.split("\n").filter((line) => /fixture|model unavailable|invalid config|failed to load plugin/i.test(line)).slice(-12).join("\n")
        throw new Error(`V2 mock host exited ${code}: ${errors}\nMock requests: ${JSON.stringify(probes)}\n${relevant}`)
    }
    return output
}

try {
    const port = await new Promise((resolve) => server.listen(0, "127.0.0.1", () => resolve(server.address().port)))
    const project = path.join(sandbox, "project")
    const configHome = path.join(sandbox, "config")
    await mkdir(project)
    await mkdir(path.join(project, ".opencode"))
    await writeFile(path.join(project, "diagnostic.txt"), Array.from({ length: 300 }, (_, index) =>
        `Historical diagnostic ${index}: ${"repeated provider-visible tool evidence ".repeat(4)}`).join("\n"))
    await mkdir(path.join(configHome, "opencode"), { recursive: true })
    await writeFile(path.join(configHome, "opencode", "better-compact.jsonc"), JSON.stringify({
        compaction: { preset: "custom", custom: { triggerPercent: 1, targetPercent: 25, recentToolTokens: 0, prefixSummary: false } },
    }))
    await writeFile(path.join(project, "opencode.jsonc"), JSON.stringify({
        $schema: "https://opencode.ai/config.json",
        model: "vllm/mock",
        plugins: [root],
        compaction: { auto: false },
        providers: { vllm: {
            package: "@opencode/ai/providers/openai-compatible",
            settings: { baseURL: `http://127.0.0.1:${port}/v1` },
            models: { mock: { name: "Fixture", package: "@opencode/ai/providers/openai-compatible",
                capabilities: { tools: true, input: ["text"], output: ["text"] },
                limit: { context: 50000, output: 2048 } } },
        } },
    }))
    const env = { ...process.env, HOME: path.join(sandbox, "home"), XDG_CONFIG_HOME: configHome,
        XDG_DATA_HOME: path.join(sandbox, "data"), XDG_CACHE_HOME: path.join(sandbox, "cache"),
        OPENCODE_DB: path.join(sandbox, "staging.db"),
        OPENCODE_PASSWORD: "fixture-only" }
    await mkdir(env.HOME)
    const executable = process.env.OPENCODE_BIN ?? "opencode"
    const apiPort = await new Promise((resolve) => {
        const socket = createTcpServer()
        socket.listen(0, "127.0.0.1", () => { const value = socket.address().port; socket.close(() => resolve(value)) })
    })
    host = spawn(executable, ["serve", "--hostname", "127.0.0.1", "--port", String(apiPort)],
        { cwd: project, env, stdio: ["ignore", "pipe", "pipe"] })
    await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => reject(new Error("V2 fixture host failed to start")), 30_000)
        host.once("error", reject)
        host.once("exit", (code) => reject(new Error(`V2 fixture host exited ${code}`)))
        host.stdout.on("data", (chunk) => { if (String(chunk).includes("server listening")) { clearTimeout(timeout); resolve() } })
    })
    const address = `http://127.0.0.1:${apiPort}`
    const created = JSON.parse(await run(executable, env, project, ["api", "--server", address, "post", "/api/session", "--data",
        '{"title":"Better Compact mock host"}']))
    const sessionID = created.data?.id
    if (!sessionID) throw new Error("V2 mock session creation did not return an ID")
    const docs = JSON.parse(await run(executable, env, project, ["api", "--server", address, "get", "/api/config"]))
    if (!docs.some((entry) => entry.info?.providers?.vllm)) throw new Error("vLLM fixture provider config was not discovered")
    let available = ""
    for (let attempt = 0; attempt < 12; attempt++) {
        available = await run(executable, env, project, ["models", "--server", address])
        if (available.includes("vllm/mock")) break
        await new Promise((resolve) => setTimeout(resolve, 250))
    }
    if (!available.includes("vllm/mock")) throw new Error(`Mock model not discovered. Probes: ${JSON.stringify(probes)}; models tail: ${available.slice(-500)}`)
    await run(executable, env, project, ["run", "--server", address, "--session", sessionID, "--model", "vllm/mock",
        "Never overwrite user files; keep the current test decision."])
    await run(executable, env, project, ["run", "--server", address, "--session", sessionID, "--model", "vllm/mock",
        "Continue the implementation without dropping the original constraint."])
    if (requests.length < 3) throw new Error(`Expected a tool loop and continuation, got ${requests.length} provider requests`)
    const second = JSON.stringify(requests[1].messages ?? [])
    const third = JSON.stringify(requests[2].messages ?? [])
    if (!second.includes("Never overwrite user files") || !second.includes("current test decision") ||
        !third.includes("Never overwrite user files") || !third.includes("Continue the implementation"))
        throw new Error("V2 provider-visible continuation dropped human intent")
    const status = JSON.parse(await run(executable, env, project, ["api", "--server", address,
        "post", "/api/rpc/better-compact/status", "--data", JSON.stringify({ input: { sessionID } })])).output
    if (status?.lastStatus !== "applied" || status.afterTokens >= status.beforeTokens || status.archiveCount < 1)
        throw new Error(`V2 context hook did not shrink the request: ${JSON.stringify(status)}`)
    const catalog = JSON.parse(await readFile(path.join(project, ".opencode", "better-compact", "v2", "sessions", sessionID, "catalog.json"), "utf8"))
    const recovered = []
    for (const entry of catalog.entries) {
        const raw = await readFile(path.join(project, ".opencode", "better-compact", "v2", "sessions", sessionID,
            "archives", `${entry.id}.json`), "utf8")
        if (createHash("sha256").update(raw).digest("hex") !== entry.sha256)
            throw new Error("V2 archive hash did not match the session catalog")
        recovered.push(...JSON.parse(raw))
    }
    const result = recovered.flatMap((message) => message.content ?? []).find((part) => part.type === "tool-result")
    if (!result || JSON.stringify(result).length < 50_000 || recovered.some((message) => message.id?.startsWith("bcv-")))
        throw new Error("V2 archive did not preserve exact ID-less native tool evidence")
    if (second.length >= 30_000 || third.length >= 30_000 ||
        second.includes("Historical diagnostic 100") || third.includes("Historical diagnostic 100"))
        throw new Error("V2 provider request still contains old tool output")
    // Run the same real host/tool loop as a control with the plugin disabled.
    // This measures the provider-visible savings rather than trusting an estimator.
    await writeFile(path.join(configHome, "opencode", "better-compact.jsonc"), JSON.stringify({ enabled: false }))
    const controlID = JSON.parse(await run(executable, env, project, ["api", "--server", address, "post", "/api/session", "--data",
        '{"title":"Better Compact disabled control"}'])).data?.id
    if (!controlID) throw new Error("V2 control session creation failed")
    await run(executable, env, project, ["run", "--server", address, "--session", controlID, "--model", "vllm/mock",
        "Never overwrite user files; keep the current test decision."])
    await run(executable, env, project, ["run", "--server", address, "--session", controlID, "--model", "vllm/mock",
        "Continue the implementation without dropping the original constraint."])
    if (requests.length !== 6) throw new Error(`Expected six mock provider calls, got ${requests.length}`)
    const baselineSecond = JSON.stringify(requests[4].messages ?? [])
    const baselineThird = JSON.stringify(requests[5].messages ?? [])
    if (!baselineSecond.includes("Historical diagnostic 100") || !baselineThird.includes("Historical diagnostic 100") ||
        second.length >= baselineSecond.length * 0.4 || third.length >= baselineThird.length * 0.4)
        throw new Error("V2 did not deliver provider-visible token savings against the disabled control")
    for (const request of [requests[1], requests[2], requests[4], requests[5]]) {
        const calls = request.messages.flatMap((message) => message.tool_calls?.map((part) => part.id) ?? [])
        const results = request.messages.filter((message) => message.role === "tool").map((message) => message.tool_call_id)
        if (JSON.stringify(calls) !== JSON.stringify(results)) throw new Error("V2 produced an orphaned native tool pair")
    }
    // Assistant-heavy history needs a validated handoff: the cheap tool ladder
    // cannot replace real implementation decisions with an arbitrary stub.
    await writeFile(path.join(configHome, "opencode", "better-compact.jsonc"), JSON.stringify({
        compaction: { preset: "custom", summaryEffort: "off", custom: {
            triggerPercent: 1, targetPercent: 10, recentToolTokens: 15000, prefixSummary: true,
        } },
    }))
    const summaryID = JSON.parse(await run(executable, env, project, ["api", "--server", address, "post", "/api/session", "--data",
        '{"title":"Better Compact validated summary"}'])).data?.id
    if (!summaryID) throw new Error("V2 summary session creation failed")
    for (let index = 0; index < 8; index++) await run(executable, env, project,
        ["run", "--server", address, "--session", summaryID, "--model", "vllm/mock",
            index ? "Continue the implementation without dropping the original constraint."
                : "Never overwrite user files; keep the current test decision."])
    const summaryStatus = JSON.parse(await run(executable, env, project, ["api", "--server", address,
        "post", "/api/rpc/better-compact/status", "--data", JSON.stringify({ input: { sessionID: summaryID } })])).output
    const summaryCatalog = JSON.parse(await readFile(path.join(project, ".opencode", "better-compact", "v2", "sessions", summaryID, "catalog.json"), "utf8"))
    if (requests.length !== 14 || summaries.length > 4 || !summaries.some((entry) => entry.kind === "handoff") ||
        !summaries.some((entry) => entry.kind === "description") ||
        !JSON.stringify(requests[13].messages).includes("## Decisions") ||
        !JSON.stringify(requests[13].messages).includes("Never overwrite user files") ||
        !JSON.stringify(requests[13].messages).includes("Continue the implementation") ||
        summaryStatus.readyCount < 1 || !summaryStatus.hasCheckpoint)
        throw new Error(`V2 validated host summary did not reach the provider: ${JSON.stringify({
            calls: requests.length, summaries: summaries.map((entry) => entry.kind), status: summaryStatus,
            providerBytes: requests.slice(6).map((request) => JSON.stringify(request.messages).length),
            hasHandoff: JSON.stringify(requests[13].messages).includes("## Decisions"),
            hasOriginal: JSON.stringify(requests[13].messages).includes("Never overwrite user files"),
            hasCurrent: JSON.stringify(requests[13].messages).includes("Continue the implementation"),
            handoffsByRequest: requests.slice(6).map((request) => JSON.stringify(request.messages).includes("## Decisions")),
            entries: summaryCatalog.entries.map((entry) => ({ sequence: entry.sequence, status: entry.status })),
            summaryAttempt: summaryCatalog.summaryAttempt,
            request13: requests[13].messages.map((message) => ({ role: message.role,
                length: JSON.stringify(message.content ?? "").length,
                sample: typeof message.content === "string" ? message.content.slice(0, 100) : "structured",
            })),
        })}`)
    await writeFile(path.join(configHome, "opencode", "better-compact.jsonc"), JSON.stringify({ enabled: false }))
    const summaryControlID = JSON.parse(await run(executable, env, project,
        ["api", "--server", address, "post", "/api/session", "--data",
            '{"title":"Better Compact assistant-heavy disabled control"}'])).data?.id
    if (!summaryControlID) throw new Error("V2 assistant-heavy control creation failed")
    for (let index = 0; index < 8; index++) await run(executable, env, project,
        ["run", "--server", address, "--session", summaryControlID, "--model", "vllm/mock",
            index ? "Continue the implementation without dropping the original constraint."
                : "Never overwrite user files; keep the current test decision."])
    const summaryBytes = JSON.stringify(requests[13].messages).length
    const fullBytes = JSON.stringify(requests[21]?.messages).length
    if (requests.length !== 22 || summaryBytes >= fullBytes * 0.9)
        throw new Error(`V2 handoff failed to reduce provider-visible assistant history: ${summaryBytes}/${fullBytes}`)
    const log = await readFile(path.join(sandbox, "data", "opencode", "log", "opencode.log"), "utf8").catch(() => "")
    if (!log.includes("loading plugin") || !log.includes("better-compact"))
        throw new Error("V2 host did not load the Better Compact plugin")
    console.log(`V2 host: ${requests.length} primary requests, ${summaries.length} side calls; tool enabled/control bytes=${second.length}/${baselineSecond.length}, ${third.length}/${baselineThird.length}; assistant-history enabled/control bytes=${summaryBytes}/${fullBytes}; exact native archive, validated handoff, and human constraints verified.`)
} finally {
    if (host && host.exitCode === null && host.signalCode === null) {
        const done = new Promise((resolve) => host.once("exit", resolve))
        host.kill()
        await done
    }
    await new Promise((resolve) => server.close(resolve))
    await rm(sandbox, { recursive: true, force: true })
}
