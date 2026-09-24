import assert from "node:assert/strict"
import test from "node:test"
import type { WithParts } from "../lib/state"
import { buildBoundaryContextPlan, formatBoundaryReport } from "../lib/boundary"

const sessionID = "ses_boundary_report"

function textPart(messageID: string, text: string) {
    return {
        id: `${messageID}-part`,
        messageID,
        sessionID,
        type: "text" as const,
        text,
    }
}

function reasoningPart(messageID: string, text: string) {
    return {
        id: `${messageID}-reasoning`,
        messageID,
        sessionID,
        type: "reasoning" as const,
        text,
        time: { start: 1 },
    }
}

function toolPart(messageID: string, tool: string, output: string) {
    return {
        id: `${messageID}-${tool}`,
        messageID,
        sessionID,
        type: "tool" as const,
        callID: `${messageID}-${tool}-call`,
        tool,
        state: {
            status: "completed" as const,
            input: tool === "skill" ? { name: "root-cause-debug" } : { filePath: "src/app.ts" },
            output,
            title: tool,
            metadata: {},
            time: { start: 1, end: 2 },
        },
    }
}

function message(
    id: string,
    role: "user" | "assistant",
    parts: WithParts["parts"],
    created: number,
): WithParts {
    return {
        info: {
            id,
            role,
            sessionID,
            agent: "assistant",
            model: { providerID: "anthropic", modelID: "claude-test" },
            time: { created },
        } as WithParts["info"],
        parts,
    }
}

test("boundary report shows visual context bars without internal threshold jargon", () => {
    const plan = buildBoundaryContextPlan(
        [
            message(
                "msg-user-1",
                "user",
                [textPart("msg-user-1", "Please preserve this exact requirement.")],
                1,
            ),
            message(
                "msg-assistant-1",
                "assistant",
                [
                    reasoningPart("msg-assistant-1", "private reasoning ".repeat(2_000)),
                    textPart("msg-assistant-1", "Investigated the OpenCode compaction path."),
                    toolPart("msg-assistant-1", "read", "tool-output ".repeat(4_000)),
                    toolPart("msg-assistant-1", "skill", "skill content ".repeat(2_000)),
                ],
                2,
            ),
            message(
                "msg-user-2",
                "user",
                [textPart("msg-user-2", "Continue with the plugin-only design.")],
                3,
            ),
            message(
                "msg-assistant-2",
                "assistant",
                [textPart("msg-assistant-2", "Recent assistant tail should remain raw.")],
                4,
            ),
            message(
                "msg-user-3",
                "user",
                [textPart("msg-user-3", "Latest user tail should remain raw.")],
                5,
            ),
        ],
        {
            contextLimit: 20_000,
            force: true,
            recentToolResultBudgetTokens: 0,
            providerReportedTokens: 18_000,
        },
    )
    assert.ok(plan)

    const report = formatBoundaryReport(plan, 18_000)

    assert.match(report, /Better Compact Complete/)
    assert.match(report, /Before\s+18K\s+\/\s+20K\s+\[/)
    assert.match(report, /Now\s+.+\[/)
    assert.match(report, /Actions/)
    assert.match(report, /Reference/)
    assert.match(report, /Target 6K \(30% best effort; not a minimum\)/)
    assert.match(report, /still above target/)
    assert.doesNotMatch(report, /Projected after/i)
    assert.doesNotMatch(report, /Trigger threshold/i)
    assert.doesNotMatch(report, /Last-resort target/i)
})

test("zero model calls never label an OpenCode prefix as a deterministic fallback", () => {
    const plan = buildBoundaryContextPlan(
        [
            message("u1", "user", [textPart("u1", "Keep current goals")], 1),
            message("a1", "assistant", [textPart("a1", "Earlier implementation ".repeat(100))], 2),
            message("u2", "user", [textPart("u2", "Continue")], 3),
            message("a2", "assistant", [textPart("a2", "Recent work")], 4),
            message("u3", "user", [textPart("u3", "Current action")], 5),
        ],
        { contextLimit: 100, force: true, prefixSummaryAllowed: true },
    )
    assert.ok(plan)
    const report = formatBoundaryReport(plan, undefined, 0)
    assert.match(
        report,
        /Luna calls: 0\. Stage changes are context deltas, not model response sizes/,
    )
    assert.equal(plan.requiresCustomCompaction, false)
    assert.doesNotMatch(report, /Deterministic prefix fallback/)
})

test("an applied summary stage with no net savings is not reported as unused", () => {
    const plan = buildBoundaryContextPlan(
        [
            message("u-1", "user", [textPart("u-1", "Old task")], 1),
            message("a-1", "assistant", [textPart("a-1", "Old reply ".repeat(1_000))], 2),
            message("u-2", "user", [textPart("u-2", "Second task")], 3),
            message("a-2", "assistant", [textPart("a-2", "Second reply")], 4),
            message("u-3", "user", [textPart("u-3", "Current task")], 5),
        ],
        { contextLimit: 40_000, force: true },
    )
    assert.ok(plan)
    const report = formatBoundaryReport({
        ...plan,
        stages: [
            ...plan.stages,
            {
                name: "assistant-runs",
                label: "Summarized assistant turns",
                status: "applied",
                clearedTokens: 0,
                changedMessages: 1,
                changedParts: 1,
                beforeTokens: 165_156,
                afterTokens: 165_156,
            },
        ],
    })
    assert.match(report, /Summarized assistant turns\s+applied \(no net savings\)/)
    assert.doesNotMatch(report, /Summarized assistant turns\s+not needed/)
})

test("report distinguishes expanding stages and unmet target from provider starting usage", () => {
    const plan = buildBoundaryContextPlan(
        [
            message("u-1", "user", [textPart("u-1", "Old task")], 1),
            message("a-1", "assistant", [textPart("a-1", "Old reply ".repeat(1_000))], 2),
            message("u-2", "user", [textPart("u-2", "New task")], 3),
            message("a-2", "assistant", [textPart("a-2", "New reply")], 4),
            message("u-3", "user", [textPart("u-3", "Latest task")], 5),
        ],
        { contextLimit: 40_000, force: true },
    )
    assert.ok(plan)
    const report = formatBoundaryReport({
        ...plan,
        beforeTokens: 20_000,
        afterPruneTokens: 15_000,
        targetTokens: 8_000,
        stages: [
            {
                name: "assistant-runs",
                label: "Summarized assistant turns",
                status: "applied",
                clearedTokens: 0,
                changedMessages: 1,
                changedParts: 1,
                beforeTokens: 30_000,
                afterTokens: 35_000,
            },
        ],
    })
    assert.match(report, /Summarized assistant turns\s+increased \+5K/)
    assert.match(report, /Target 8K \(20% best effort; not a minimum\)/)
    assert.match(report, /7K still above target/)
    assert.match(report, /30K local raw estimate; Before is provider-reported/)
})

test("above-target report identifies the largest final retained components", () => {
    const plan = buildBoundaryContextPlan(
        [
            message("u-1", "user", [textPart("u-1", "Older instruction")], 1),
            message("a-1", "assistant", [textPart("a-1", "Old work ".repeat(400))], 2),
            message("u-2", "user", [textPart("u-2", "Continue")], 3),
            message("a-2", "assistant", [textPart("a-2", "Recent work ".repeat(100))], 4),
            message("u-3", "user", [textPart("u-3", "Current work")], 5),
        ],
        { contextLimit: 20_000, force: true, targetTokens: 1, archiveCatalogText: "" },
    )
    assert.ok(plan)
    assert.equal(plan.requiresCustomCompaction, false)
    assert.ok(plan.residual)
    const report = formatBoundaryReport(plan)
    assert.match(report, /still above target/)
    assert.match(report, /Largest retained components: /)
    assert.match(report, /recent raw tail|handoff\/reference and archive catalog/)
    assert.doesNotMatch(report, /Largest retained components: .*undefined/)
})

test("report explains a two-output reasoning shortfall without adding a second reserve", () => {
    const plan = buildBoundaryContextPlan(
        [
            message("u-old", "user", [textPart("u-old", "Older request")], 1),
            message("a-old", "assistant", [textPart("a-old", "Prior work ".repeat(100))], 2),
            message("u-now", "user", [textPart("u-now", "Current request")], 3),
        ],
        { contextLimit: 12_000, force: true, minTailUserTurns: 1 },
    )
    assert.ok(plan)
    const report = formatBoundaryReport({
        ...plan,
        anchoredOutputCount: 2,
        recentAssistantOutputs: 5,
        recentReasoningBudgetTokens: 28_000,
        anchorReasoningLimited: true,
        anchorReasoningFloorUnmet: true,
    })
    assert.match(report, /one bounded reasoning allowance \(up to 28K when it fits\)/)
    assert.match(report, /could not fit the 20K reasoning minimum beside 2 outputs/)
})
