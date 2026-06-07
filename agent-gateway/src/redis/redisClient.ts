import { createClient, type RedisClientType } from "redis";

export type RedisLike = {
  get(key: string): Promise<string | null>;
  set(key: string, value: string): Promise<unknown>;
  setEx(key: string, ttl: number, value: string): Promise<unknown>;
  expire(key: string, ttl: number): Promise<unknown>;
  rPush(key: string, value: string): Promise<unknown>;
  lTrim(key: string, start: number, stop: number): Promise<unknown>;
  zAdd(key: string, entries: Array<{ score: number; value: string }>): Promise<unknown>;
  ping(): Promise<unknown>;
  quit?(): Promise<unknown>;
  del?(key: string): Promise<unknown>;
};

export async function connectRedis(url: string): Promise<RedisClientType> {
  const client = createClient({ url });
  client.on("error", (error) => {
    console.error("[redis]", url, error);
  });
  await client.connect();
  return client as RedisClientType;
}
