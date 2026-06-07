export function safeJsonParse<T = unknown>(value: string | null | undefined): T | null {
  if (!value) return null;
  try {
    return JSON.parse(value) as T;
  } catch {
    return null;
  }
}

export function ndjson(value: unknown): string {
  return `${JSON.stringify(value)}\n`;
}

export function truncate(text: string, max = 220): string {
  return text.length > max ? text.slice(0, max).trimEnd() : text;
}
