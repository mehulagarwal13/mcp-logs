import { expect, type APIRequestContext } from "@playwright/test";

/**
 * Email verification needs a real, delivered email. The e2e backend is
 * configured (`.github/workflows/e2e-and-eval.yml`) to submit mail to a
 * Mailpit SMTP sink; this reads the message back through Mailpit's HTTP API
 * and extracts the verification token from the link, exactly as a person would
 * click it. When `E2E_MAIL_API_URL` is unset (no sink available) the specs
 * that need a delivered email are skipped, not faked.
 *
 * The token only ever exists here, in the test process's memory.
 */
export const MAIL_API_URL = process.env.E2E_MAIL_API_URL ?? "";
export const MAIL_AVAILABLE = MAIL_API_URL !== "";
export const MAIL_SKIP_REASON =
  "needs a mail sink: set E2E_MAIL_API_URL (Mailpit HTTP API) and start the backend with EMAIL_PROVIDER=smtp pointing at it";

interface MailpitSearch {
  messages?: { ID: string }[];
}
interface MailpitMessage {
  Text?: string;
}

/** Polls the sink until a verification email addressed to `email` arrives and
 * returns the token from its `/verify-email#token=...` link. */
export async function waitForVerificationToken(
  request: APIRequestContext,
  email: string,
  { timeoutMs = 30_000 }: { timeoutMs?: number } = {},
): Promise<string> {
  const deadline = Date.now() + timeoutMs;
  while (Date.now() < deadline) {
    const search = await request.get(`${MAIL_API_URL}/api/v1/search`, {
      params: { query: `to:${email}` },
    });
    if (search.ok()) {
      const { messages = [] } = (await search.json()) as MailpitSearch;
      for (const { ID } of messages) {
        const message = (await (await request.get(`${MAIL_API_URL}/api/v1/message/${ID}`)).json()) as MailpitMessage;
        const match = message.Text?.match(/verify-email#token=([A-Za-z0-9_-]+)/);
        if (match) return match[1];
      }
    }
    await new Promise((resolve) => setTimeout(resolve, 500));
  }
  expect.soft(false, `no verification email for ${email} within ${timeoutMs}ms`).toBeTruthy();
  throw new Error(`no verification email for ${email}`);
}
