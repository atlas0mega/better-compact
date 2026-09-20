import assert from "node:assert/strict"
import test from "node:test"
import { resolveCompactionVariant } from "../lib/boundary/summarizer"

const params = { providerId: "local", modelId: "qwen", agent: undefined, variant: "high" }
const client = { provider: { list: async () => ({ data: { all: [
    { id: "local", models: { qwen: { variants: { low: {}, high: {}, xhigh: {} } } } },
] } }) } }

test("compaction reasoning uses the requested model variant", async () => {
    assert.equal(await resolveCompactionVariant(client, params, "low"), "low")
    assert.equal(await resolveCompactionVariant(client, params, "max"), "xhigh")
    assert.equal(params.variant, "high")
})

test("inherit, unsupported effort and provider errors preserve the active variant", async () => {
    assert.equal(await resolveCompactionVariant(client, params, "inherit"), "high")
    assert.equal(await resolveCompactionVariant(client, params, "medium"), "high")
    assert.equal(await resolveCompactionVariant({}, params, "low"), "high")
    assert.equal(await resolveCompactionVariant(client, { ...params, modelId: "other" }, "low"), "high")
})
