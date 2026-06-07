import type { AuthUser } from "../types.js";
import { safeJsonParse } from "../utils/json.js";
import type { RedisLike } from "./redisClient.js";

export const AUTH_SESSION_PREFIX = "robotagent:auth:session";

export function authSessionKey(token: string): string {
  return `${AUTH_SESSION_PREFIX}:${token}`;
}

export function extractBearerToken(authorization: string | undefined): string | null {
  const header = (authorization || "").trim();
  if (!header.toLowerCase().startsWith("bearer ")) return null;
  const token = header.slice(7).trim();
  return token || null;
}

export async function requireAuthUser(
  redis: RedisLike,
  authorization: string | undefined,
  ttlSeconds: number
): Promise<AuthUser> {
  const token = extractBearerToken(authorization);
  if (!token) {
    throw Object.assign(new Error("缺少认证令牌"), { statusCode: 401 });
  }

  const raw = await redis.get(authSessionKey(token));
  if (!raw) {
    throw Object.assign(new Error("登录已过期，请重新登录"), { statusCode: 401 });
  }

  const session = safeJsonParse<Record<string, unknown>>(raw);
  if (!session || typeof session !== "object") {
    throw Object.assign(new Error("登录会话无效"), { statusCode: 401 });
  }

  await redis.expire(authSessionKey(token), ttlSeconds);
  return {
    token,
    uid: String(session.uid || "unknown"),
    username: session.username ? String(session.username) : undefined
  };
}
