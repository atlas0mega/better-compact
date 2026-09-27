# OpenCode V2 package checks

From `packages/opencode`:

- `pnpm check:package` builds the V2 server, TUI and RPC entrypoints and checks
  their exports, dependencies, tarball contents and absence of a V1 entrypoint.
- `pnpm smoke:install` packs and installs the tarball in an isolated temporary
  project, then verifies that a private OpenCode V2 server loads it as active.
- `pnpm verify:release -- v0.3.0` checks a release tag against `package.json`.

Session inspection uses OpenCode V2's `opencode api` and the plugin's scoped
`better_compact_recall` tool/RPC rather than direct database compatibility
scripts. These checks never enable the plugin in the user's global config.
