import { apiRequest, mockDelay } from "./client";
import { USE_MOCK_DATA } from "./config";
import type { McpToolDefinition, McpToolUsage } from "@/types/mcp";
import { mockMcpToolCatalogue, mockMcpToolUsage } from "@/mocks/data/mcp";

/**
 * The tools the MCP server has registered, with descriptions and parameters
 * (`GET /observability/mcp/tools`). In mock mode this is the snapshot generated
 * from the real registration (`scripts/export_mcp_tool_catalogue.py`).
 */
export async function listMcpToolCatalogue(): Promise<McpToolDefinition[]> {
  if (USE_MOCK_DATA) {
    return mockDelay(mockMcpToolCatalogue);
  }
  return apiRequest<McpToolDefinition[]>(`/observability/mcp/tools`);
}

/**
 * Recorded usage of MCP tools by the caller's organization
 * (`GET /observability/mcp`). Statistics only -- not the catalogue.
 */
export async function listMcpToolUsage(): Promise<McpToolUsage[]> {
  if (USE_MOCK_DATA) {
    return mockDelay(mockMcpToolUsage);
  }
  return apiRequest<McpToolUsage[]>(`/observability/mcp`);
}
