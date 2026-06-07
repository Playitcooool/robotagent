export class FakeRedis {
  values = new Map<string, string>();
  lists = new Map<string, string[]>();
  zsets = new Map<string, Map<string, number>>();
  expirations = new Map<string, number>();

  async get(key: string): Promise<string | null> {
    return this.values.get(key) || null;
  }

  async set(key: string, value: string): Promise<string> {
    this.values.set(key, value);
    return "OK";
  }

  async setEx(key: string, ttl: number, value: string): Promise<string> {
    this.values.set(key, value);
    this.expirations.set(key, ttl);
    return "OK";
  }

  async expire(key: string, ttl: number): Promise<boolean> {
    this.expirations.set(key, ttl);
    return true;
  }

  async rPush(key: string, value: string): Promise<number> {
    const list = this.lists.get(key) || [];
    list.push(value);
    this.lists.set(key, list);
    return list.length;
  }

  async lTrim(key: string, start: number, stop: number): Promise<string> {
    const list = this.lists.get(key) || [];
    const normalizedStart = start < 0 ? Math.max(list.length + start, 0) : start;
    const normalizedStop = stop < 0 ? list.length + stop : stop;
    this.lists.set(key, list.slice(normalizedStart, normalizedStop + 1));
    return "OK";
  }

  async zAdd(key: string, entries: Array<{ score: number; value: string }>): Promise<number> {
    const zset = this.zsets.get(key) || new Map<string, number>();
    for (const entry of entries) zset.set(entry.value, entry.score);
    this.zsets.set(key, zset);
    return entries.length;
  }

  async ping(): Promise<string> {
    return "PONG";
  }

  async quit(): Promise<string> {
    return "OK";
  }
}
