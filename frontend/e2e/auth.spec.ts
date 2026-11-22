import { test, expect } from "@playwright/test";
import { loginViaUI, signupViaUI } from "./fixtures/actions";
import { MAIL_AVAILABLE, MAIL_SKIP_REASON, waitForVerificationToken } from "./fixtures/mail";
import { ORG_ALPHA } from "./fixtures/testUsers";

test.describe("Authentication", () => {
  test("an unauthenticated visitor hitting a protected route is redirected to login", async ({ page }) => {
    await page.goto("/dashboard");
    await expect(page).toHaveURL(/\/login$/);
    await expect(page.getByRole("heading", { name: "Sign in to EKIP" })).toBeVisible();
  });

  test("signup creates nothing and shows a check-your-email screen, not the app", async ({ page }) => {
    test.skip(!MAIL_AVAILABLE, MAIL_SKIP_REASON);
    const unique = Date.now();
    await signupViaUI(page, {
      name: "E2E Fresh Signup",
      email: `e2e-fresh-signup-${unique}@example.com`,
      password: "FreshSignup123!",
      orgName: `E2E Fresh Org ${unique}`,
      orgSlug: `e2e-fresh-org-${unique}`,
    });
    // Not authenticated: no session exists until the emailed link is opened.
    await expect(page).toHaveURL(/\/signup$/);
    await expect(page.getByRole("link", { name: "Connectors" })).toHaveCount(0);
  });

  test("an unverified signup cannot sign in; the emailed link verifies it, then sign-in works", async ({
    page,
    request,
  }) => {
    test.skip(!MAIL_AVAILABLE, MAIL_SKIP_REASON);
    const unique = Date.now();
    const email = `e2e-verify-flow-${unique}@example.com`;
    const password = "FreshSignup123!";
    await signupViaUI(page, {
      name: "E2E Verify Flow",
      email,
      password,
      orgName: `E2E Verify Org ${unique}`,
      orgSlug: `e2e-verify-org-${unique}`,
    });

    // Before the link is opened the account does not exist: generic failure.
    await page.goto("/login");
    await page.getByLabel("Work email").fill(email);
    await page.getByLabel("Password").fill(password);
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByText("Sign in failed")).toBeVisible();

    // Open the emailed link (token in the URL fragment).
    const token = await waitForVerificationToken(request, email);
    await page.goto(`/verify-email#token=${token}`);
    await expect(page.getByRole("heading", { name: "Email confirmed" })).toBeVisible({ timeout: 30_000 });
    // The token is removed from the address bar immediately and never stored.
    await expect(page).toHaveURL(/\/verify-email$/);
    expect(await page.evaluate(() => JSON.stringify({ ...localStorage, ...sessionStorage }))).not.toContain(token);

    // Verification returned no session: signing in is still required...
    await page.getByRole("button", { name: "Go to sign in" }).click();
    await page.getByLabel("Work email").fill(email);
    await page.getByLabel("Password").fill(password);
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page).toHaveURL(/\/ask$/, { timeout: 30_000 });
    // ...and the new admin has the real permission set from `GET /auth/me`.
    await expect(page.getByRole("link", { name: "Connectors" })).toBeVisible();

    // A second use of the same link is rejected.
    await page.goto(`/verify-email#token=${token}`);
    await expect(page.getByRole("heading", { name: "This link was already used" })).toBeVisible();
  });

  test("the verify page rejects a made-up token without creating anything", async ({ page }) => {
    await page.goto("/verify-email#token=this-is-not-a-real-token");
    await expect(page.getByRole("heading", { name: "This confirmation link isn’t valid" })).toBeVisible({
      timeout: 15_000,
    });
  });

  test("the verify page with no token explains the link is incomplete", async ({ page }) => {
    await page.goto("/verify-email");
    await expect(page.getByRole("heading", { name: "This confirmation link is incomplete" })).toBeVisible();
  });

  test("login with valid credentials reaches the authenticated app", async ({ page }) => {
    await loginViaUI(page, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    await expect(page).toHaveURL(/\/ask$/);
    await expect(page.getByRole("button", { name: `Account menu for ${ORG_ALPHA.admin.name}` })).toBeVisible();
  });

  test("login with a wrong password fails without authenticating", async ({ page }) => {
    await page.goto("/login");
    await page.getByLabel("Work email").fill(ORG_ALPHA.admin.email);
    await page.getByLabel("Password").fill("definitely-wrong-password");
    await page.getByRole("button", { name: "Sign in" }).click();
    await expect(page.getByText("Sign in failed")).toBeVisible();
    await expect(page).toHaveURL(/\/login$/);
  });

  test("a 401 from an authenticated API call clears the session and returns to login", async ({ page }) => {
    // core.auth.service issues short-lived (jwt_expiry_minutes) access
    // tokens, so waiting for a real one to expire is impractical for a
    // test. This intercepts one real, already-authenticated request and
    // substitutes a 401 EKIPError body -- the exact shape `app.api.errors.
    // ekip_error_handler` produces -- to deterministically exercise
    // `api/client.ts`'s real `clearSessionAndNotifyExpired` path (the
    // Phase 1 fix) rather than waiting out a token lifetime. Everything
    // downstream (AuthContext clearing `user`, ProtectedRoute redirecting)
    // is the genuine app reacting to that response, not simulated.
    await loginViaUI(page, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);

    await page.route("**/observability/agents", (route) =>
      route.fulfill({
        status: 401,
        contentType: "application/json",
        body: JSON.stringify({ error_code: "auth.invalid_token", message: "Invalid or expired access token." }),
      }),
    );
    await page.goto("/agents");

    await expect(page).toHaveURL(/\/login$/, { timeout: 10_000 });
  });
});
