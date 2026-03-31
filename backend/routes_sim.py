import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.responses import FileResponse, StreamingResponse


def register_sim_routes(
    app,
    *,
    sim_stream_dir: Path,
    sim_meta_file: Path,
    sim_frame_file: Path,
    sim_frame_cache: dict[str, Any],
):
    def load_latest_frame_payload():
        if not sim_meta_file.exists() or not sim_frame_file.exists():
            return {"status": "idle", "has_frame": False}

        try:
            meta_mtime = sim_meta_file.stat().st_mtime
            frame_mtime = sim_frame_file.stat().st_mtime
        except Exception:
            meta_mtime = 0.0
            frame_mtime = 0.0

        cached = sim_frame_cache
        if meta_mtime == cached.get("meta_mtime") and frame_mtime == cached.get(
            "frame_mtime"
        ):
            return cached.get("payload", {"status": "idle", "has_frame": False})

        try:
            with open(sim_meta_file, "r", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception as e:
            return {
                "status": "error",
                "has_frame": False,
                "error": f"meta read failed: {e}",
            }

        payload = {
            "status": "done" if meta.get("done") else "running",
            "has_frame": True,
            "run_id": meta.get("run_id"),
            "task": meta.get("task"),
            "step": meta.get("step"),
            "total_steps": meta.get("total_steps"),
            "done": bool(meta.get("done")),
            "timestamp": meta.get("timestamp"),
            "image_url": f"/api/sim/latest.png?ts={meta.get('timestamp')}",
        }
        sim_frame_cache["meta_mtime"] = meta_mtime
        sim_frame_cache["frame_mtime"] = frame_mtime
        sim_frame_cache["payload"] = payload
        return payload

    @app.get("/api/sim/debug")
    async def sim_debug():
        return {
            "stream_dir": str(sim_stream_dir),
            "meta_exists": sim_meta_file.exists(),
            "frame_exists": sim_frame_file.exists(),
        }

    @app.get("/api/sim/latest-frame")
    async def get_latest_sim_frame():
        return load_latest_frame_payload()

    @app.get("/api/sim/latest.png")
    async def get_latest_sim_png():
        if not sim_frame_file.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="frame not found",
            )
        return FileResponse(
            sim_frame_file,
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/api/sim/stream")
    async def stream_sim_frames(request: Request, since: float = 0.0):
        async def event_stream():
            last_ts = float(since or 0.0)
            idle_ticks = 0
            last_emit_ts = 0.0
            initial_sent = False

            while True:
                if await request.is_disconnected():
                    break

                payload = load_latest_frame_payload()
                if payload.get("has_frame"):
                    current_ts = float(payload.get("timestamp") or 0.0)
                    if not initial_sent:
                        initial_sent = True
                        last_ts = max(last_ts, current_ts)
                        idle_ticks = 0
                        last_emit_ts = time.time()
                        yield (
                            f"event: frame\\ndata: "
                            f"{json.dumps(payload, ensure_ascii=False)}\\n\\n"
                        )
                    elif current_ts > last_ts:
                        last_ts = current_ts
                        idle_ticks = 0
                        last_emit_ts = time.time()
                        yield (
                            f"event: frame\\ndata: "
                            f"{json.dumps(payload, ensure_ascii=False)}\\n\\n"
                        )
                    else:
                        idle_ticks += 1
                else:
                    idle_ticks += 1

                if idle_ticks >= 100:
                    idle_ticks = 0
                    yield "event: ping\\ndata: {}\\n\\n"

                now = time.time()
                is_running = payload.get("status") == "running"
                if is_running and now - last_emit_ts < 2.0:
                    sleep_s = 0.05
                elif idle_ticks > 200:
                    sleep_s = 0.5
                else:
                    sleep_s = 0.2
                await asyncio.sleep(sleep_s)

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache, no-transform",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
