import { test, expect } from "@playwright/test";
import { apiLogin, decodeOrganizationId, API_BASE_URL } from "./fixtures/actions";
import { MAIL_AVAILABLE, MAIL_SKIP_REASON, waitForVerificationToken } from "./fixtures/mail";
import { ORG_ALPHA } from "./fixtures/testUsers";

/**
 * Exercises the invitation flow end-to-end under email verification:
 * `POST /organizations/{id}/invitations` returns a raw token exactly once (to the
 * ADMIN, so it is not proof of mailbox control); `POST /invitations/{id}/accept`
 * (`core.auth.service.accept_invitation_with_password`) only starts acceptance and
 * emails a confirmation link; the user, role and invitation consumption happen when
 * that link is opened. This spec drives invitation creation
 * via the real REST API (no UI for it exists to seed from otherwise -- see
 * `UsersSettingsPage`'s "Invite user" modal for the one place a human would
 * do this) and drives *acceptance* through the actual `AcceptInvitationPage`
 * UI, since that's the surface being verified.
 */
test.describe("Accept invitation", () => {
  test.skip(!MAIL_AVAILABLE, MAIL_SKIP_REASON);

  test("accepting only starts verification; the invitee exists once they confirm the emailed link", async ({
    page,
    request,
  }) => {
    const accessToken = await apiLogin(request, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    const organizationId = decodeOrganizationId(accessToken);
    const email = `e2e-invitee-${Date.now()}@example.com`;

    const createResponse = await request.post(`${API_BASE_URL}/organizations/${organizationId}/invitations`, {
      headers: { Authorization: `Bearer ${accessToken}` },
      data: { email, grants_role: "member" },
    });
    expect(createResponse.ok(), `create_invitation failed: ${createResponse.status()}`).toBeTruthy();
    const invitation = await createResponse.json();
    expect(invitation.token).toBeTruthy();

    await page.goto(`/invitations/${invitation.id}/accept?token=${invitation.token}`);
    await page.getByLabel("Your name").fill("E2E Invitee");
    await page.getByLabel("Choose a password").fill("AcceptedInvite123!");
    await page.getByRole("button", { name: "Accept invitation" }).click();

    // The invitation token is visible to the inviting admin, so it is NOT proof
    // of mailbox control: no session, no user yet -- a confirmation email first.
    await expect(page.getByRole("heading", { name: "Check your email" })).toBeVisible({ timeout: 15_000 });
    const loginBefore = await request.post(`${API_BASE_URL}/auth/login`, {
      data: { email, password: "AcceptedInvite123!" },
    });
    expect(loginBefore.status()).toBe(403);

    const token = await waitForVerificationToken(request, email);
    await page.goto(`/verify-email#token=${token}`);
    await expect(page.getByRole("heading", { name: "Email confirmed" })).toBeVisible({ timeout: 30_000 });

    // Now a real, distinct session exists for the newly provisioned user.
    const invitee = await apiLogin(request, email, "AcceptedInvite123!");
    const meResponse = await request.get(`${API_BASE_URL}/auth/me`, {
      headers: { Authorization: `Bearer ${invitee}` },
    });
    expect(meResponse.ok()).toBeTruthy();
    expect((await meResponse.json()).email).toBe(email);
  });

  test("an invitation for an address that already has an account cannot take it over", async ({ request }) => {
    const accessToken = await apiLogin(request, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    const organizationId = decodeOrganizationId(accessToken);

    // Invite the (existing) restricted user, then try to "accept" with an attacker-chosen password.
    const createResponse = await request.post(`${API_BASE_URL}/organizations/${organizationId}/invitations`, {
      headers: { Authorization: `Bearer ${accessToken}` },
      data: { email: ORG_ALPHA.restricted.email, grants_role: "member" },
    });
    test.skip(!createResponse.ok(), "restricted user already has a pending invitation in this seed");
    const invitation = await createResponse.json();

    const accept = await request.post(`${API_BASE_URL}/invitations/${invitation.id}/accept`, {
      data: { token: invitation.token, password: "attacker-chosen-Password1!" },
    });
    expect(accept.status()).toBe(409);
    // The victim's password is untouched...
    const stillWorks = await request.post(`${API_BASE_URL}/auth/login`, {
      data: { email: ORG_ALPHA.restricted.email, password: ORG_ALPHA.restricted.password },
    });
    expect(stillWorks.ok()).toBeTruthy();
    // ...and the attacker's is not accepted.
    const attackerLogin = await request.post(`${API_BASE_URL}/auth/login`, {
      data: { email: ORG_ALPHA.restricted.email, password: "attacker-chosen-Password1!" },
    });
    expect(attackerLogin.ok()).toBeFalsy();
  });

  test("an invitation that was only started (not confirmed) is not consumed", async ({ page, request }) => {
    const accessToken = await apiLogin(request, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    const organizationId = decodeOrganizationId(accessToken);
    const email = `e2e-invitee-pending-${Date.now()}@example.com`;
    const createResponse = await request.post(`${API_BASE_URL}/organizations/${organizationId}/invitations`, {
      headers: { Authorization: `Bearer ${accessToken}` },
      data: { email, grants_role: "member" },
    });
    const invitation = await createResponse.json();

    const first = await request.post(`${API_BASE_URL}/invitations/${invitation.id}/accept`, {
      data: { token: invitation.token, password: "AcceptedInvite123!" },
    });
    expect(first.status()).toBe(202);
    expect(await first.text()).not.toContain("access_token");

    // Still pending: the same link can be used again until the mailbox is proven.
    await page.goto(`/invitations/${invitation.id}/accept?token=${invitation.token}`);
    await page.getByLabel("Choose a password").fill("AnotherPassword123!");
    await page.getByRole("button", { name: "Accept invitation" }).click();
    await expect(page.getByRole("heading", { name: "Check your email" })).toBeVisible({ timeout: 15_000 });
  });
});
