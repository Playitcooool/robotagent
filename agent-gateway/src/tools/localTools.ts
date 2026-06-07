import fs from "node:fs/promises";
import path from "node:path";
import { XMLParser } from "fast-xml-parser";
import type { GatewayConfig, ToolDefinition } from "../types.js";

function schema(properties: Record<string, unknown>, required: string[] = []): Record<string, unknown> {
  return { type: "object", properties, required };
}

function resolveRepoPath(rootDir: string, inputPath: string): string {
  const candidate = path.resolve(rootDir, inputPath || ".");
  const relative = path.relative(rootDir, candidate);
  if (relative.startsWith("..") || path.isAbsolute(relative)) {
    throw new Error("access denied: path outside repository root");
  }
  return candidate;
}

async function walk(root: string, dir: string, max: number, files: string[] = []): Promise<string[]> {
  if (files.length >= max) return files;
  const entries = await fs.readdir(dir, { withFileTypes: true });
  for (const entry of entries) {
    if (entry.name === "node_modules" || entry.name === ".git" || entry.name === ".venv") continue;
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) await walk(root, full, max, files);
    else if (entry.isFile()) files.push(path.relative(root, full));
    if (files.length >= max) break;
  }
  return files;
}

function globMatch(file: string, globPattern: string): boolean {
  if (!globPattern || globPattern === "**/*") return true;
  const escaped = globPattern
    .replace(/[.+^${}()|[\]\\]/g, "\\$&")
    .replace(/\*\*/g, ".*")
    .replace(/\*/g, "[^/]*");
  return new RegExp(`^${escaped}$`).test(file);
}

async function tavilySearch(config: GatewayConfig, query: string, maxResults: number, timeout: number): Promise<Record<string, unknown>> {
  if (!config.tavilyApiKey) return { results: [], answer: "", warning: "TAVILY_API_KEY not set" };
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout * 1000);
  try {
    const response = await fetch("https://api.tavily.com/search", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        api_key: config.tavilyApiKey,
        query,
        max_results: maxResults,
        include_answer: true,
        include_raw_content: false
      }),
      signal: controller.signal
    });
    if (!response.ok) throw new Error(`Tavily returned ${response.status}`);
    return (await response.json()) as Record<string, unknown>;
  } finally {
    clearTimeout(timer);
  }
}

async function webSearch(config: GatewayConfig, args: Record<string, unknown>): Promise<string> {
  const query = String(args.query || "").trim();
  if (!query) throw new Error("query is required");
  const maxResults = Math.max(1, Math.min(Number(args.max_results || 5), 10));
  const timeout = Math.max(1, Number(args.timeout || 8));
  const raw = await tavilySearch(config, query, maxResults, timeout);
  const items = Array.isArray(raw.results) ? raw.results : [];
  const results = items.slice(0, maxResults).map((item) => {
    const row = item && typeof item === "object" ? (item as Record<string, unknown>) : {};
    return {
      title: String(row.title || "Unknown"),
      url: String(row.url || ""),
      snippet: String(row.content || row.snippet || ""),
      source: "tavily"
    };
  });
  return JSON.stringify({ query, engine: "tavily", returned: results.length, results, answer: raw.answer || "" }, null, 2);
}

async function academicSearch(args: Record<string, unknown>): Promise<string> {
  const query = String(args.query || "").trim();
  if (!query) throw new Error("query is required");
  const maxResults = Math.max(1, Math.min(Number(args.max_results || 5), 20));
  const params = new URLSearchParams({
    search_query: `ti:${query}+OR+abs:${query}`,
    start: "0",
    max_results: String(maxResults),
    sortBy: "relevance"
  });
  const response = await fetch(`https://export.arxiv.org/api/query?${params.toString()}`);
  if (!response.ok) throw new Error(`arXiv returned ${response.status}`);
  const xml = await response.text();
  const parser = new XMLParser({ ignoreAttributes: false });
  const parsed = parser.parse(xml) as Record<string, unknown>;
  const feed = parsed.feed && typeof parsed.feed === "object" ? (parsed.feed as Record<string, unknown>) : {};
  const entries = Array.isArray(feed.entry) ? feed.entry : feed.entry ? [feed.entry] : [];
  const results = entries.slice(0, maxResults).map((entry) => {
    const item = entry as Record<string, unknown>;
    const links = Array.isArray(item.link) ? item.link : item.link ? [item.link] : [];
    const pdfLink = links.find((link) => link && typeof link === "object" && (link as Record<string, unknown>)["@_title"] === "pdf") as Record<string, unknown> | undefined;
    const pdfUrl = String(pdfLink?.["@_href"] || "");
    const arxivId = pdfUrl.split("/").pop()?.replace(".pdf", "") || "";
    const authors = Array.isArray(item.author) ? item.author : item.author ? [item.author] : [];
    return {
      type: "paper",
      title: String(item.title || "Unknown").replace(/\s+/g, " ").trim(),
      authors: authors.map((author) => String((author as Record<string, unknown>).name || "")).filter(Boolean).slice(0, 3).join(", "),
      year: String(item.published || "").slice(0, 4),
      venue: "arXiv",
      abstract: String(item.summary || "").replace(/\s+/g, " ").trim().slice(0, 900),
      url: arxivId ? `https://arxiv.org/abs/${arxivId}` : "",
      pdf_url: pdfUrl,
      source: "arxiv"
    };
  });
  return JSON.stringify({ query, returned: results.length, results }, null, 2);
}

export function createLocalTools(config: GatewayConfig): ToolDefinition[] {
  return [
    {
      name: "current_time",
      description: "Return current timestamp in ISO format for a timezone.",
      parameters: schema({ tz: { type: "string", default: "UTC" } }),
      execute: async (args) => new Date().toLocaleString("sv-SE", { timeZone: String(args.tz || "UTC") })
    },
    {
      name: "list_workspace_files",
      description: "List files under the repository workspace safely.",
      parameters: schema({ glob_pattern: { type: "string", default: "**/*" }, max_results: { type: "number", default: 200 } }),
      execute: async (args) => {
        const max = Math.max(1, Math.min(Number(args.max_results || 200), 2000));
        const files = (await walk(config.rootDir, config.rootDir, max * 3)).filter((file) => globMatch(file, String(args.glob_pattern || "**/*"))).slice(0, max);
        return JSON.stringify({ root: config.rootDir, glob_pattern: args.glob_pattern || "**/*", count: files.length, files, truncated: files.length >= max }, null, 2);
      }
    },
    {
      name: "read_workspace_file",
      description: "Read a text file under the repository workspace safely.",
      parameters: schema({ path: { type: "string" }, max_chars: { type: "number", default: 12000 } }, ["path"]),
      execute: async (args) => {
        const candidate = resolveRepoPath(config.rootDir, String(args.path || ""));
        const content = await fs.readFile(candidate, "utf8");
        const max = Math.max(1, Number(args.max_chars || 12000));
        return JSON.stringify({ path: path.relative(config.rootDir, candidate), content: content.slice(0, max), truncated: content.length > max }, null, 2);
      }
    },
    {
      name: "search_workspace_text",
      description: "Search for plain text in workspace files and return matched lines.",
      parameters: schema({ pattern: { type: "string" }, glob_pattern: { type: "string", default: "**/*" }, max_results: { type: "number", default: 100 } }, ["pattern"]),
      execute: async (args) => {
        const pattern = String(args.pattern || "");
        if (!pattern) throw new Error("pattern is required");
        const max = Math.max(1, Math.min(Number(args.max_results || 100), 1000));
        const files = (await walk(config.rootDir, config.rootDir, 5000)).filter((file) => globMatch(file, String(args.glob_pattern || "**/*")));
        const matches: Record<string, unknown>[] = [];
        for (const file of files) {
          const full = path.join(config.rootDir, file);
          const content = await fs.readFile(full, "utf8").catch(() => "");
          content.split(/\r?\n/).forEach((line, index) => {
            if (matches.length < max && line.includes(pattern)) matches.push({ path: file, line: index + 1, text: line.slice(0, 300) });
          });
          if (matches.length >= max) break;
        }
        return JSON.stringify({ pattern, count: matches.length, matches, truncated: matches.length >= max }, null, 2);
      }
    },
    {
      name: "http_get",
      description: "Perform a safe HTTP GET request and return status, headers, and body snippet.",
      parameters: schema({ url: { type: "string" }, max_chars: { type: "number", default: 3000 }, timeout: { type: "number", default: 8 } }, ["url"]),
      execute: async (args) => {
        const url = String(args.url || "");
        if (!/^https?:\/\//i.test(url)) throw new Error("only http/https URLs are allowed");
        const response = await fetch(url, { signal: AbortSignal.timeout(Math.max(1, Number(args.timeout || 8)) * 1000) });
        const body = await response.text();
        const max = Math.max(1, Number(args.max_chars || 3000));
        const headers: Record<string, string> = {};
        response.headers.forEach((value, key) => {
          headers[key] = value;
        });
        return JSON.stringify({ url, status_code: response.status, headers, body_snippet: body.slice(0, max), truncated: body.length > max }, null, 2);
      }
    },
    {
      name: "web_search",
      description: "Search the web for recent/general information using Tavily.",
      parameters: schema({ query: { type: "string" }, max_results: { type: "number", default: 5 }, timeout: { type: "number", default: 8 } }, ["query"]),
      execute: (args) => webSearch(config, args)
    },
    {
      name: "academic_search",
      description: "Search academic papers from arXiv.",
      parameters: schema({ query: { type: "string" }, max_results: { type: "number", default: 5 }, timeout: { type: "number", default: 15 } }, ["query"]),
      execute: academicSearch
    },
    {
      name: "search",
      description: "Unified search tool combining web and academic search results.",
      parameters: schema({ query: { type: "string" }, max_results: { type: "number", default: 5 }, timeout: { type: "number", default: 30 } }, ["query"]),
      execute: async (args) => {
        const [web, academic] = await Promise.allSettled([webSearch(config, args), academicSearch(args)]);
        const webResults = web.status === "fulfilled" ? JSON.parse(web.value).results || [] : [];
        const academicResults = academic.status === "fulfilled" ? JSON.parse(academic.value).results || [] : [];
        const limit = Math.max(1, Math.min(Number(args.max_results || 5), 20));
        return JSON.stringify({ query: String(args.query || ""), engine: "web + academic", returned: webResults.length + academicResults.length, results: [...webResults, ...academicResults].slice(0, limit) }, null, 2);
      }
    },
    {
      name: "format_json",
      description: "Validate and pretty-format JSON input.",
      parameters: schema({ json_str: { type: "string" }, sort_keys: { type: "boolean", default: false } }, ["json_str"]),
      execute: async (args) => JSON.stringify(JSON.parse(String(args.json_str || "")), null, 2)
    }
  ];
}
