import { describe, expect, it } from "vitest";
import { authSessionKey, requireAuthUser } from "../src/redis/auth.js";
import { appendChatMessage, chatHistoryKey, chatSessionsKey, getOrCreatePiSessionId, piSessionKey } from "../src/redis/chatHistory.js";
import { FakeRedis } from "./fakes.js";

describe("redis compatibility", () => {
  it("validates auth session and refreshes ttl", async () => {
    const redis = new FakeRedis();
    await redis.set(authSessionKey("token"), JSON.stringify({ uid: "u1", username: "alice" }));
    const user = await requireAuthUser(redis, "Bearer token", 60);
    expect(user).toEqual({ token: "token", uid: "u1", username: "alice" });
    expect(redis.expirations.get(authSessionKey("token"))).toBe(60);
  });

  it("writes chat history in Python-compatible format", async () => {
    const redis = new FakeRedis();
    await appendChatMessage(redis, "u1", "s1", "user", "hello", 200);
    const list = redis.lists.get(chatHistoryKey("u1", "s1")) || [];
    expect(list).toHaveLength(1);
    expect(JSON.parse(list[0])).toMatchObject({ role: "user", text: "hello", session_id: "s1" });
    expect(redis.zsets.get(chatSessionsKey("u1"))?.has("s1")).toBe(true);
  });

  it("persists Pi session mapping", async () => {
    const redis = new FakeRedis();
    const first = await getOrCreatePiSessionId(redis, "u1", "s1");
    const second = await getOrCreatePiSessionId(redis, "u1", "s1");
    expect(first).toBe(second);
    expect(redis.values.get(piSessionKey("u1", "s1"))).toBe(first);
  });
});
