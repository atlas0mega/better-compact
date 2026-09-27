import { builtinModules, createRequire } from "node:module"
import { existsSync, readFileSync, readdirSync, statSync } from "node:fs"
import { execFileSync } from "node:child_process"
import path from "node:path"
import process from "node:process"
import { fileURLToPath, pathToFileURL } from "node:url"

const require = createRequire(import.meta.url)
const root = path.dirname(path.dirname(fileURLToPath(import.meta.url)))

const builtinNames = new Set([
    ...builtinModules,
    ...builtinModules.map((name) => name.replace(/^node:/, "")),
])

const allowedNamedImportPackages = new Set(["jsonc-parser"])

const requiredRepoFiles = [
    "index.js",
    "dist/index.js",
    "dist/tui.js",
    "dist/index.d.ts",
    "dist/tui.d.ts",
    "dist/rpc.js",
    "dist/rpc.d.ts",
    "better-compact.schema.json",
    "README.md",
    "LICENSE",
]

const requiredTarballFiles = [
    "package.json",
    "index.js",
    "dist/index.js",
    "dist/tui.js",
    "dist/index.d.ts",
    "dist/tui.d.ts",
    "dist/rpc.js",
    "dist/rpc.d.ts",
    "better-compact.schema.json",
    "README.md",
    "LICENSE",
]

const forbiddenTarballPatterns = [
    /^node_modules\//,
    /^index\.ts$/,
    /^lib\//,
    /^dist\/lib\//,
    /\.map$/,
    /^tests\//,
    /^scripts\//,
    /^docs\//,
    /^assets\//,
    /^notes\//,
    /^\.github\//,
    /^package-lock\.json$/,
    /^tsconfig\.json$/,
]

const packageInfoCache = new Map()

function fail(message) {
    console.error(`package verification failed: ${message}`)
    process.exit(1)
}

function assertRepoFilesExist() {
    for (const relativePath of requiredRepoFiles) {
        if (!existsSync(path.join(root, relativePath))) {
            fail(`missing required file: ${relativePath}`)
        }
    }
}

function assertNoUnpublishedChunks() {
    const chunks = readdirSync(path.join(root, "dist")).filter(
        (name) => name.endsWith(".js") && !["index.js", "tui.js", "rpc.js"].includes(name),
    )
    if (chunks.length) fail(`build emitted JS chunks not included in the published tarball: ${chunks.join(", ")}`)
}

function assertPackageJsonShape() {
    const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"))

    if (pkg.main !== "./dist/index.js") {
        fail(`package.json main must remain ./dist/index.js, found ${pkg.main ?? "<missing>"}`)
    }

    if (pkg.exports?.["."]?.import !== "./dist/index.js") {
        fail("expected package.json exports['.'].import to be './dist/index.js'")
    }

    if (pkg.exports?.["./tui"]?.import !== "./dist/tui.js") {
        fail("expected package.json exports['./tui'].import to be './dist/tui.js'")
    }
    if (pkg.exports?.["./rpc"]?.import !== "./dist/rpc.js") {
        fail("expected package.json exports['./rpc'].import to be './dist/rpc.js'")
    }
    if (pkg.exports?.["./server"] || pkg.dependencies?.["@opencode-ai/plugin"] ||
        pkg.devDependencies?.["@opencode-ai/sdk"]) {
        fail("V1 package surface remains")
    }
    if (pkg.engines?.opencode !== ">=2.0.18 <3") {
        fail("package.json engines.opencode must be '>=2.0.18 <3'")
    }

    const files = Array.isArray(pkg.files) ? pkg.files : []
    for (const entry of [
        "index.js",
        "dist/index.js",
        "dist/index.d.ts",
        "dist/tui.js",
        "dist/tui.d.ts",
        "dist/rpc.js",
        "dist/rpc.d.ts",
        "better-compact.schema.json",
        "README.md",
        "LICENSE",
    ]) {
        if (!files.includes(entry)) {
            fail(`package.json files must include ${entry}`)
        }
    }

    if (files.includes("lib/") || files.includes("dist/")) {
        fail("package.json files must explicitly allowlist runtime entrypoints")
    }
}

function getPublishedBareImports() {
    const specifiers = new Set()
    for (const entry of ["dist/index.js", "dist/tui.js", "dist/rpc.js"]) {
        const source = readFileSync(path.join(root, entry), "utf8")
        for (const [, specifier] of source.matchAll(/\bfrom\s*["']([^"']+)["']/g)) {
            if (specifier.startsWith(".") || specifier.startsWith("/")) continue
            const name = getPackageName(specifier)
            if (!builtinNames.has(name)) specifiers.add(name)
        }
    }
    return specifiers
}

function assertInstallableDependencies() {
    const pkg = JSON.parse(readFileSync(path.join(root, "package.json"), "utf8"))
    const peers = pkg.peerDependencies ?? {}
    const dependencies = pkg.dependencies ?? {}
    const meta = pkg.peerDependenciesMeta ?? {}
    if (dependencies["@opencode/plugin"] !== "2.0.18" || dependencies["@opencode/ai"] !== "2.0.18")
        fail("V2 runtime dependencies must be explicitly installable at the tested versions")

    const imported = getPublishedBareImports()
    if (imported.size === 0) {
        fail("found no bare imports in the built bundles; the import scan is no longer working")
    }
    for (const specifier of imported) {
        if (!(specifier in peers) && !(specifier in dependencies)) {
            fail(`dist imports ${specifier}, so package.json must declare a runtime dependency`)
        }
    }

    for (const [name, range] of Object.entries(peers)) {
        if (!meta[name]?.optional) continue
        if (!/^>=\d+\.\d+\.\d+$/.test(range)) {
            fail(
                `peer dependency ${name} is host-provided, so its range must be a bare lower bound like '>=0.4.2', found '${range}'`,
            )
        }
    }
}

async function validateBuiltEntrypoints() {
    const server = await import(pathToFileURL(path.join(root, "dist/index.js")).href)
    if (server.default?.id !== "better-compact" || typeof server.default.setup !== "function" || "server" in server.default) {
        fail("dist/index.js must default export a V2-only plugin definition")
    }

    const tui = await import(pathToFileURL(path.join(root, "dist/tui.js")).href)
    if (
        !tui.default ||
        tui.default.id !== "better-compact.tui" ||
        typeof tui.default.setup !== "function" || "tui" in tui.default
    ) {
        fail("dist/tui.js must default export a V2-only TUI plugin")
    }
    const rpc = await import(pathToFileURL(path.join(root, "dist/rpc.js")).href)
    if (rpc.BetterCompactRPC?.id !== "better-compact") fail("dist/rpc.js must export the V2 RPC contract")
}

function getImportStatements(source) {
    const pattern = /^\s*import\s+([^\n;]+?)\s+from\s+["']([^"']+)["']/gm
    const imported = Array.from(source.matchAll(pattern), (match) => ({
        clause: match[1].trim(),
        specifier: match[2],
    }))
    const exports = Array.from(source.matchAll(/^\s*export\s+([^\n;]+?)\s+from\s+["']([^"']+)["']/gm),
        (match) => ({ clause: match[1].trim(), specifier: match[2] }))
    return [...imported, ...exports]
}

function getImportKind(clause) {
    if (clause.startsWith("type ")) return "type"
    if (clause.startsWith("* as ")) return "namespace"
    if (clause.startsWith("{")) return "named"
    if (clause.includes(",")) {
        const [, trailing = ""] = clause.split(",", 2)
        return trailing.trim().startsWith("* as ") ? "default+namespace" : "default+named"
    }
    return "default"
}

function getPackageName(specifier) {
    if (specifier.startsWith("@")) {
        const parts = specifier.split("/")
        return parts.length >= 2 ? `${parts[0]}/${parts[1]}` : specifier
    }
    return specifier.split("/")[0]
}

function resolveLocalImport(importerPath, specifier) {
    const basePath = path.resolve(path.dirname(importerPath), specifier)
    const candidates = [
        basePath,
        `${basePath}.ts`,
        `${basePath}.tsx`,
        `${basePath}.js`,
        `${basePath}.mjs`,
        path.join(basePath, "index.ts"),
        path.join(basePath, "index.tsx"),
        path.join(basePath, "index.js"),
        path.join(basePath, "index.mjs"),
    ]

    for (const candidate of candidates) {
        if (existsSync(candidate) && statSync(candidate).isFile()) return candidate
    }

    fail(`unable to resolve local import ${specifier} from ${path.relative(root, importerPath)}`)
}

function findPackageInfo(packageName, importerPath) {
    const cacheKey = `${packageName}::${path.dirname(importerPath)}`
    if (packageInfoCache.has(cacheKey)) {
        return packageInfoCache.get(cacheKey)
    }

    let entry
    try {
        entry = require.resolve(packageName, { paths: [path.dirname(importerPath)] })
    } catch {
        packageInfoCache.set(cacheKey, null)
        return null
    }

    let current = path.dirname(entry)
    while (true) {
        const manifest = path.join(current, "package.json")
        if (existsSync(manifest)) {
            const info = JSON.parse(readFileSync(manifest, "utf8"))
            packageInfoCache.set(cacheKey, info)
            return info
        }
        const parent = path.dirname(current)
        if (parent === current) {
            packageInfoCache.set(cacheKey, null)
            return null
        }
        current = parent
    }
}

function packageLooksCommonJs(pkg) {
    if (!pkg) return false
    if (pkg.type === "commonjs") return true

    const main = typeof pkg.main === "string" ? pkg.main : ""
    return /(?:^|\/)(cjs|umd)(?:\/|$)/.test(main) || main.endsWith(".cjs")
}

function validateRuntimeImportGraph() {
    const pending = [path.join(root, "index.ts"), path.join(root, "tui.ts"), path.join(root, "rpc.ts")]
    const seen = new Set()

    while (pending.length > 0) {
        const filePath = pending.pop()
        if (!filePath || seen.has(filePath)) continue
        seen.add(filePath)

        const source = readFileSync(filePath, "utf8")
        for (const entry of getImportStatements(source)) {
            if (entry.specifier.startsWith(".")) {
                pending.push(resolveLocalImport(filePath, entry.specifier))
                continue
            }

            if (entry.specifier === "jsonc-parser/lib/esm/main.js") {
                continue
            }

            const packageName = getPackageName(entry.specifier)
            if (builtinNames.has(packageName)) continue

            const kind = getImportKind(entry.clause)
            if (kind === "type" || kind === "namespace") continue
            if (allowedNamedImportPackages.has(packageName)) continue

            const pkg = findPackageInfo(packageName, filePath)
            if (packageLooksCommonJs(pkg)) {
                fail(
                    `${path.relative(root, filePath)} uses ${kind} import from CommonJS-style package ${packageName}`,
                )
            }
        }
    }
}

function assertNoV1Source() {
    const paths = ["index.ts", "tui.ts", "rpc.ts", "index.js"]
    const walk = (directory) => {
        for (const entry of readdirSync(path.join(root, directory), { withFileTypes: true })) {
            const relative = path.join(directory, entry.name)
            if (entry.isDirectory()) walk(relative)
            else if (/\.(?:ts|tsx|js)$/.test(entry.name)) paths.push(relative)
        }
    }
    walk("lib")
    for (const relative of paths) {
        const source = readFileSync(path.join(root, relative), "utf8")
        if (/@opencode-ai\/(?:plugin|sdk)|experimental\.chat\.|\bPluginInput\b|\.client\.tui\./.test(source))
            fail(`V1 plugin implementation remains in ${relative}`)
    }
}

function validatePackedFiles() {
    const output = execFileSync("npm", ["pack", "--dry-run", "--json"], {
        cwd: root,
        encoding: "utf8",
    })

    const metadata = JSON.parse(output)
    // npm versions emit either an array or an object keyed by package name.
    const [result] = Array.isArray(metadata) ? metadata : Object.values(metadata)
    if (!result || !Array.isArray(result.files)) {
        fail("npm pack --dry-run --json did not return file metadata")
    }

    const packedPaths = result.files.map((file) => file.path)
    if (packedPaths.length !== requiredTarballFiles.length) {
        fail(
            `packed tarball contains ${packedPaths.length} files; expected ${requiredTarballFiles.length}`,
        )
    }
    for (const required of requiredTarballFiles) {
        if (!packedPaths.includes(required)) {
            fail(`packed tarball is missing ${required}`)
        }
    }

    const forbidden = packedPaths.find((file) =>
        forbiddenTarballPatterns.some((pattern) => pattern.test(file)),
    )
    if (forbidden) {
        fail(`packed tarball contains forbidden path ${forbidden}`)
    }

    console.log(`package verification passed for ${result.name}@${result.version}`)
    console.log(`tarball entries: ${result.entryCount}`)
}

assertRepoFilesExist()
assertNoUnpublishedChunks()
assertPackageJsonShape()
assertInstallableDependencies()
validateRuntimeImportGraph()
assertNoV1Source()
await validateBuiltEntrypoints()
validatePackedFiles()
