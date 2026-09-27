import { defineConfig } from "tsup"
import { version } from "./package.json"

const define = { __BC_VERSION__: JSON.stringify(version) }

export default defineConfig([
    {
        entry: { index: "index.ts" },
        format: ["esm"],
        splitting: false,
        dts: true,
        clean: true,
        sourcemap: false,
        define,
        external: ["@opencode/ai", "@opencode/plugin", "@opencode/plugin/rpc", "@opencode/plugin/tui"],
        noExternal: ["@better-compact/core", "jsonc-parser"],
    },
    {
        entry: { tui: "tui.ts", rpc: "rpc.ts" },
        format: ["esm"],
        splitting: false,
        dts: true,
        clean: false,
        sourcemap: false,
        external: ["@opencode/plugin", "@opencode/plugin/tui", "@opencode/plugin/rpc"],
        define,
        noExternal: ["@better-compact/core", "jsonc-parser"],
    },
])
