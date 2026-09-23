import assert from "node:assert/strict"
import test from "node:test"
import { getLastUserMessage, isIgnoredUserMessage } from "../lib/messages/query"
import { isSyndicatePluginInjection } from "../lib/messages/injection"
import type { WithParts } from "../lib/state"

function buildMessage(role: "user" | "assistant", parts: WithParts["parts"]): WithParts {
    const sessionID = "ses_message_utils"

    const info =
        role === "user"
            ? {
                  id: `msg-${role}`,
                  role,
                  sessionID,
                  agent: "assistant",
                  model: {
                      providerID: "anthropic",
                      modelID: "claude-test",
                  },
                  time: { created: 1 },
              }
            : {
                  id: `msg-${role}`,
                  role,
                  sessionID,
                  agent: "assistant",
                  time: { created: 1 },
              }

    return {
        info: info as WithParts["info"],
        parts,
    }
}

test("isIgnoredUserMessage only ignores user messages", () => {
    const ignoredUserMessage = buildMessage("user", [])
    const assistantMessage = buildMessage("assistant", [])

    assert.equal(isIgnoredUserMessage(ignoredUserMessage), true)
    assert.equal(isIgnoredUserMessage(assistantMessage), false)
})

test("only user-role text prompts with the Syndicate plugin provenance suffix are generated", () => {
    const suffix = "\n\n[plugin-injection:12345678-1234-4234-8234-123456789abc]"
    const injection = buildMessage("user", [
        {
            type: "text",
            id: "part-plugin",
            sessionID: "ses_message_utils",
            messageID: "msg-user",
            text: `[teams-md] An activation prompt${suffix}`,
        },
    ])
    const untaggedInjection = buildMessage("user", [
        {
            type: "text",
            id: "part-untagged",
            sessionID: "ses_message_utils",
            messageID: "msg-user",
            text: `An untagged plugin alert${suffix}`,
        },
    ])
    assert.equal(isSyndicatePluginInjection(injection), true)
    assert.equal(isSyndicatePluginInjection(untaggedInjection), true)
    assert.equal(isIgnoredUserMessage(injection), false)
    const human = buildMessage("user", [
        {
            type: "text",
            id: "part-human",
            sessionID: "ses_message_utils",
            messageID: "msg-user",
            text: "My actual instruction",
        },
    ])
    // The latest injected message still supplies the active specialist's
    // agent/model tuple; the codec separately excludes it as user intent.
    assert.equal(getLastUserMessage([human, injection]), injection)
    assert.equal(
        isSyndicatePluginInjection({ ...injection, info: buildMessage("assistant", []).info }),
        false,
    )
    assert.equal(
        isSyndicatePluginInjection({
            ...injection,
            parts: [{ ...injection.parts[0], text: `${suffix}\nmore user text` }],
        }),
        false,
    )
    assert.equal(
        isSyndicatePluginInjection({
            ...injection,
            parts: [{ ...injection.parts[0], text: "[plugin-injection:not-a-uuid]" }],
        }),
        false,
    )
    assert.equal(
        isSyndicatePluginInjection({
            ...injection,
            parts: [{ ...injection.parts[0], ignored: true }],
        }),
        false,
    )
    assert.equal(isIgnoredUserMessage(human), false)
})
