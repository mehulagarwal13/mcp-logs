import type { McpToolDefinition, McpToolParameter, McpToolUsage } from "@/types/mcp";
import { minutesAgo } from "@/mocks/time";
import catalogueSnapshot from "./mcpToolCatalogue.generated.json";

/**
 * Mock-mode catalogue. NOT hand-written: `mcpToolCatalogue.generated.json` is
 * produced from the MCP server's real tool registration by
 * `scripts/export_mcp_tool_catalogue.py`, and a backend test fails if it drifts
 * from that registration. Only `status`/`syncedAt` are filled in here.
 */
interface CatalogueSnapshotEntry {
  name: string;
  description: string;
  parameters: McpToolParameter[];
}

export const mockMcpToolCatalogue: McpToolDefinition[] = (
  catalogueSnapshot as CatalogueSnapshotEntry[]
).map((tool) => ({
  name: tool.name,
  description: tool.description,
  status: "available",
  parameters: tool.parameters,
  syncedAt: minutesAgo(5),
}));

/**
 * Illustrative usage numbers (clearly separate from the catalogue). Names are real
 * registered tools, plus a resource entry to show activity that is not a tool.
 */
export const mockMcpToolUsage: McpToolUsage[] = [
  { toolName: "ask_question", requestCount: 1892, errorCount: 12, avgLatencyMs: 2400, maxLatencyMs: 9100 },
  { toolName: "search_similar_incidents", requestCount: 734, errorCount: 3, avgLatencyMs: 180, maxLatencyMs: 640 },
  { toolName: "investigate_incident", requestCount: 211, errorCount: 9, avgLatencyMs: 14500, maxLatencyMs: 48000 },
  { toolName: "resource:incident", requestCount: 512, errorCount: 0, avgLatencyMs: 65, maxLatencyMs: 210 },
];
