import type { ISODateString } from "./common";

export type McpParamType = "string" | "number" | "boolean" | "object" | "array";

/** One input parameter, derived by the backend from the tool's JSON Schema. */
export interface McpToolParameter {
  name: string;
  type: McpParamType;
  required: boolean;
  description: string;
  /** Display text of the schema default ("10", "true", "oidc"); null when none. */
  default: string | null;
}

/**
 * A tool the MCP server has actually REGISTERED -- `GET /observability/mcp/tools`
 * (`core.observability.schemas.McpToolDefinition`). This is the catalogue: what
 * exists and how to call it. It carries no usage numbers.
 *
 * The raw JSON Schema (`inputSchema`) is deliberately not modelled: this client's
 * snake_case -> camelCase conversion would rewrite its property names. The
 * `parameters` list is derived from that schema by the backend, with values intact.
 */
export interface McpToolDefinition {
  name: string;
  description: string;
  /** "available" = present in the MCP server's latest published registration. */
  status: "available";
  parameters: McpToolParameter[];
  /** When the MCP server last published its registration. */
  syncedAt: ISODateString;
}

/**
 * Recorded usage of one tool by THIS organization -- `GET /observability/mcp`
 * (`McpToolStats`). Statistics about past calls, not part of the catalogue: it
 * can include entries that are not tools (e.g. `resource:document`) and omits
 * tools that were never called.
 */
export interface McpToolUsage {
  toolName: string;
  requestCount: number;
  errorCount: number;
  avgLatencyMs: number | null;
  maxLatencyMs: number | null;
}
