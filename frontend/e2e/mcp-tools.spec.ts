import { test, expect } from "@playwright/test";
import { API_BASE_URL, apiLogin, loginViaUI } from "./fixtures/actions";
import { ORG_ALPHA, ORG_BETA } from "./fixtures/testUsers";

/**
 * The MCP Tools page shows the tools the MCP server has REGISTERED (published by
 * the server at startup), with their parameters -- and shows recorded usage
 * separately. Needs the MCP server to have been started against the same database
 * so it has published its catalogue (`scripts/run_mcp_server.py`).
 */
test.describe("MCP Tools catalogue", () => {
  test("lists the registered tools with descriptions and required/optional parameters", async ({
    page,
    request,
  }) => {
    // The page must match what the backend's catalogue endpoint really returns.
    const token = await apiLogin(request, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    const response = await request.get(`${API_BASE_URL}/observability/mcp/tools`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(response.ok()).toBeTruthy();
    const catalogue = (await response.json()) as Array<{
      name: string;
      description: string;
      parameters: Array<{ name: string; required: boolean }>;
    }>;
    expect(catalogue.length).toBeGreaterThan(0);

    await loginViaUI(page, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    await page.goto("/mcp");
    await expect(page.getByRole("heading", { name: /Registered tools/ })).toBeVisible();

    for (const tool of catalogue) {
      const card = page.getByTestId(`mcp-tool-${tool.name}`);
      await expect(card).toBeVisible();
      await expect(card.getByText(tool.description.split("\n")[0].trim(), { exact: false })).toBeVisible();
      for (const param of tool.parameters) {
        // Match the parameter's own name cell (exact), not any row that merely contains
        // the text -- a parameter can be called "description", like a column header.
        const row = card
          .getByRole("row")
          .filter({ has: page.getByRole("cell", { name: param.name, exact: true }) });
        await expect(row).toContainText(param.required ? "required" : "optional");
      }
    }
  });

  test("keeps usage statistics visibly separate from the catalogue", async ({ page }) => {
    await loginViaUI(page, ORG_ALPHA.admin.email, ORG_ALPHA.admin.password);
    await page.goto("/mcp");
    const card = page.getByTestId("mcp-tool-create_project");
    await expect(card.getByLabel("Recorded usage of create_project")).toBeVisible();
    await expect(card.getByText("Recorded usage:")).toBeVisible();
  });

  test("a user without observability:read cannot read the catalogue endpoint", async ({ request }) => {
    const token = await apiLogin(request, ORG_ALPHA.restricted.email, ORG_ALPHA.restricted.password);
    const response = await request.get(`${API_BASE_URL}/observability/mcp/tools`, {
      headers: { Authorization: `Bearer ${token}` },
    });
    expect(response.status()).toBe(403);
    const unauthenticated = await request.get(`${API_BASE_URL}/observability/mcp/tools`);
    expect(unauthenticated.status()).toBe(403);
  });

  test("both organizations' admins see the same registered tools", async ({ request }) => {
    const names = async (email: string, password: string) => {
      const token = await apiLogin(request, email, password);
      const response = await request.get(`${API_BASE_URL}/observability/mcp/tools`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      return ((await response.json()) as Array<{ name: string }>).map((t) => t.name);
    };
    expect(await names(ORG_BETA.admin.email, ORG_BETA.admin.password)).toEqual(
      await names(ORG_ALPHA.admin.email, ORG_ALPHA.admin.password),
    );
  });
});
