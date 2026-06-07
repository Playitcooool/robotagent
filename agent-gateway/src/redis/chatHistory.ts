import type { ChatHistoryMessage } from "../types.js";
import type { RedisLike } from "./redisClient.js";

export const CHAT_HISTORY_PREFIX = "robotagent:chat:messages";
export const CHAT_SESSIONS_ZSET_PREFIX = "robotagent:chat:sessions";
export const PI_SESSION_PREFIX = "robotagent:pi:session";

export function chatHistoryKey(userId: string, sessionId: string): string {
  const safe = (sessionId || "default_session").trim() || "default_session";
  return `${CHAT_HISTORY_PREFIX}:${userId}:${safe}`;
}

export function chatSessionsKey(userId: string): string {
  return `${CHAT_SESSIONS_ZSET_PREFIX}:${userId}`;
}

export function piSessionKey(userId: string, sessionId: string): string {
  const safe = (sessionId || "default_session").trim() || "default_session";
  return `${PI_SESSION_PREFIX}:${userId}:${safe}`;
}

export async function appendChatMessage(
  redis: RedisLike,
  userId: string,
  sessionId: string,
  role: "user" | "assistant",
  text: string,
  maxLen: number
): Promise<ChatHistoryMessage> {
  const now = Date.now() / 1000;
  const payload: ChatHistoryMessage = {
    id: Math.floor(now * 1000),
    role,
    text: text || "",
    session_id: sessionId,
    created_at: now
  };

  await redis.rPush(chatHistoryKey(userId, sessionId), JSON.stringify(payload));
  await redis.lTrim(chatHistoryKey(userId, sessionId), -maxLen, -1);
  await redis.zAdd(chatSessionsKey(userId), [{ score: now, value: sessionId }]);
  return payload;
}

export async function getOrCreatePiSessionId(redis: RedisLike, userId: string, sessionId: string): Promise<string> {
  const key = piSessionKey(userId, sessionId);
  const existing = await redis.get(key);
  if (existing) return existing;
  const mapped = `robotagent-${userId}-${sessionId}-${Date.now().toString(36)}`;
  await redis.set(key, mapped);
  return mapped;
}
