import type { BoundaryContextPlan } from "@better-compact/core"

export function formatBoundaryReport(
    plan: BoundaryContextPlan,
    actualCurrentTokens?: number,
    summaryCalls?: number,
): string {
    const before =
        actualCurrentTokens && actualCurrentTokens > 0 ? actualCurrentTokens : plan.beforeTokens
    const now = plan.afterPruneTokens
    const reduced = Math.max(0, before - now)
    const reductionPercent = before > 0 ? Math.round((reduced / before) * 100) : 0
    const rawEstimate = plan.stages[0]?.beforeTokens
    const residual = plan.residual
    const retained = residual
        ? ([
              ["recent raw tail", residual.rawTailTokens],
              ["protected reasoning/tools", residual.protectedPartTokens],
              ["handoff/reference and archive catalog", residual.handoffTokens],
              ["other retained content", residual.otherTokens],
              ["provider overhead estimate", residual.overheadTokens],
          ] as const)
        : []
    const largest = [...retained]
        .filter(([, tokens]) => tokens > 0)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3)
    const stageRows = plan.stages
        .filter((stage) => stage.status !== "skipped" || stage.clearedTokens > 0)
        .map((stage) => {
            const icon =
                stage.clearedTokens > 0 ||
                stage.status === "applied" ||
                stage.status === "target-met"
                    ? "✓"
                    : "-"
            const detail =
                stage.afterTokens > stage.beforeTokens
                    ? `increased +${formatTokenCount(stage.afterTokens - stage.beforeTokens)}`
                    : stage.clearedTokens > 0
                      ? `-${formatTokenCount(stage.clearedTokens)}`
                      : stage.status === "applied"
                        ? "applied (no net savings)"
                        : stage.status === "failed"
                          ? "did not reach target"
                          : "not needed"
            const label =
                stage.name === "prefix-summary" && summaryCalls === 0
                    ? "Deterministic prefix fallback"
                    : stage.label
            return `  ${icon} ${label.padEnd(38)} ${detail}`
        })
    return [
        "╭─────────────────────────────────────────────────────────────────────────╮",
        "│                       Better Compact Complete                          │",
        "╰─────────────────────────────────────────────────────────────────────────╯",
        "",
        "  Context window",
        formatContextWindowLine("Before", before, plan.contextLimit),
        formatContextWindowLine("Now", now, plan.contextLimit, "projected"),
        "",
        `  Reduced ${formatTokenCount(reduced)} (${reductionPercent}%)`,
        now > plan.targetTokens
            ? `  Target ${formatTokenCount(plan.targetTokens)}; ${formatTokenCount(now - plan.targetTokens)} still above target`
            : undefined,
        now > plan.targetTokens && largest.length > 0
            ? `  Largest retained components: ${largest.map(([label, tokens]) => `${label} ~${formatTokenCount(tokens)}`).join(", ")}`
            : undefined,
        plan.anchorReasoningLimited
            ? `  Five-output reasoning span exceeded the safe window; kept outputs and bounded reasoning to at most ${formatTokenCount(plan.recentReasoningBudgetTokens ?? 0)}. Exact history remains in the private archive.`
            : undefined,
        "",
        "  Actions",
        ...(stageRows.length > 0 ? stageRows : ["  (no stages applied)"]),
        rawEstimate !== undefined && rawEstimate !== before
            ? `  Stage changes use a ${formatTokenCount(rawEstimate)} local raw estimate; Before is provider-reported.`
            : undefined,
        summaryCalls !== undefined
            ? `  Luna calls: ${summaryCalls}. Stage changes are context deltas, not model response sizes.`
            : undefined,
        "",
        "  Reference",
        `  ${plan.transcript.relativePath}`,
        `  ${plan.transcript.messageIds.length} archived messages; raw tail starts at ${plan.rawTailStartMessageId}`,
        Object.keys(plan.assistantSummaries).length > 0
            ? `  ${Object.keys(plan.assistantSummaries).length} assistant summaries accepted`
            : undefined,
        "",
        "  Note: Now is Better Compact's projected active-history size. OpenCode's footer",
        "  updates after the next provider response and also includes system/tool schemas,",
        "  cache accounting, output, reasoning, and provider overhead.",
    ]
        .filter((line): line is string => line !== undefined)
        .join("\n")
}

function formatContextWindowLine(
    label: string,
    tokens: number,
    limit: number,
    suffix?: string,
): string {
    const width = 18
    const boundedLimit = Math.max(1, limit)
    const ratio = tokens / boundedLimit
    const filled = Math.max(0, Math.min(width, Math.round(Math.min(1, ratio) * width)))
    const percent = Math.round(ratio * 100)
    const suffixText = suffix ? ` ${suffix}` : ""
    return `  ${label.padEnd(6)} ${formatTokenCount(tokens).padStart(7)} / ${formatTokenCount(boundedLimit).padEnd(7)} [${"█".repeat(filled)}${"░".repeat(width - filled)}] ${String(percent).padStart(3)}%${suffixText}`
}

function formatTokenCount(tokens: number): string {
    if (tokens >= 1_000_000) return `${(tokens / 1_000_000).toFixed(1).replace(/\.0$/, "")}M`
    if (tokens >= 1000) return `${(tokens / 1000).toFixed(1).replace(/\.0$/, "")}K`
    return `${tokens}`
}
