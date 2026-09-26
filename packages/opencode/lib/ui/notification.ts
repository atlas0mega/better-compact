import type { Logger } from "../logger"

export async function sendIgnoredMessage(
    client: any,
    sessionID: string,
    text: string,
    params: any,
    logger: Logger,
): Promise<void> {
    const agent = params.agent || undefined
    const variant = params.variant || undefined
    const model =
        params.providerId && params.modelId
            ? {
                  providerID: params.providerId,
                  modelID: params.modelId,
              }
            : undefined

    // The v1 session.get response does not contain model or variant. Without
    // an observed chat-hook variant, a no-reply prompt could reset xhigh to
    // the model default. Use a non-message notification instead.
    if (!variant) {
        try {
            await client.tui.showToast({
                body: { title: "Better Compact", message: text, variant: "info", duration: 7000 },
            })
        } catch (error) {
            logger.warn("Could not show Better Compact notification", {
                error: error instanceof Error ? error.message : String(error),
            })
        }
        return
    }

    try {
        await client.session.prompt({
            path: {
                id: sessionID,
            },
            body: {
                noReply: true,
                agent: agent,
                model: model,
                variant: variant,
                parts: [
                    {
                        type: "text",
                        text: text,
                        ignored: true,
                    },
                ],
            },
        })
    } catch (error: any) {
        logger.error("Failed to send notification", { error: error.message })
    }
}
