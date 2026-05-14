import asyncio
import json
import time
from pathlib import Path
from typing import Any

from fastapi import HTTPException, Request, status
from fastapi.encoders import jsonable_encoder
from fastapi.responses import FileResponse, StreamingResponse


def register_sim_routes(
    app,
    *,
    sim_stream_dir: Path,
    sim_meta_file: Path,
    sim_frame_file: Path,
    sim_replay_dir: Path,
    sim_frame_cache: dict[str, Any],
):
    mcp_camera_lock = asyncio.Lock()
    mcp_client_state: dict[str, Any] = {"url": None, "client": None, "entered": False}

    def resolve_pybullet_mcp_url() -> str:
        import yaml as _yaml

        config_path = Path(__file__).resolve().parent.parent / "config" / "config.yml"
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = _yaml.load(f.read(), _yaml.FullLoader)
        mcp_cfg = cfg.get("mcp") or {}
        base = str(mcp_cfg.get("ip") or "http://127.0.0.1").rstrip("/")
        port = str(mcp_cfg.get("port") or "18001")
        return f"{base}:{port}/mcp"

    async def call_mcp_tool(tool_name: str, payload: dict[str, Any] | None = None):
        from fastmcp import Client

        url = resolve_pybullet_mcp_url()
        client = Client(url)
        async with client:
            try:
                return await client.call_tool(tool_name, payload or {})
            except Exception:
                if payload and "args" not in payload:
                    return await client.call_tool(tool_name, {"args": payload})
                raise

    async def call_camera_mcp_tool(payload: dict[str, Any]):
        from fastmcp import Client

        started = time.perf_counter()
        async with mcp_camera_lock:
            url = resolve_pybullet_mcp_url()
            client = mcp_client_state.get("client")
            if client is None or mcp_client_state.get("url") != url or not mcp_client_state.get("entered"):
                client = Client(url)
                await client.__aenter__()
                mcp_client_state.update({"url": url, "client": client, "entered": True})
            try:
                result = await client.call_tool("set_camera_view", payload)
            except Exception:
                if "args" not in payload:
                    result = await client.call_tool("set_camera_view", {"args": payload})
                else:
                    raise
            elapsed = time.perf_counter() - started
            print(f"[routes_sim] tool=set_camera_view total_s={elapsed:.4f}")
            return result

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
            "camera": meta.get("camera"),
        }
        sim_frame_cache["meta_mtime"] = meta_mtime
        sim_frame_cache["frame_mtime"] = frame_mtime
        sim_frame_cache["payload"] = payload
        return payload

    def load_replay_payload():
        frames: list[dict[str, Any]] = []
        if not sim_replay_dir.exists():
            return {"ok": True, "frames": frames}
        for meta_path in sorted(sim_replay_dir.glob("frame_*.json")):
            frame_id = meta_path.stem
            frame_path = sim_replay_dir / f"{frame_id}.png"
            if not frame_path.exists():
                continue
            try:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f)
            except Exception:
                meta = {}
            timestamp = float(meta.get("timestamp") or frame_path.stat().st_mtime)
            frames.append({
                "frame_id": frame_id,
                "run_id": meta.get("run_id"),
                "task": meta.get("task"),
                "step": meta.get("step"),
                "total_steps": meta.get("total_steps"),
                "done": bool(meta.get("done")),
                "timestamp": timestamp,
                "image_url": f"/api/sim/replay/{frame_id}.png?ts={timestamp}",
            })
        frames.sort(key=lambda item: float(item.get("timestamp") or 0.0))
        return {"ok": True, "frames": frames}

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

    @app.get("/api/sim/replay")
    async def get_sim_replay():
        return load_replay_payload()

    @app.get("/api/sim/replay/{frame_id}.png")
    async def get_sim_replay_png(frame_id: str):
        if not frame_id.startswith("frame_") or not frame_id.replace("frame_", "", 1).isdigit():
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="invalid frame id",
            )
        frame_path = sim_replay_dir / f"{frame_id}.png"
        if not frame_path.exists():
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="replay frame not found",
            )
        return FileResponse(
            frame_path,
            media_type="image/png",
            headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
        )

    @app.get("/api/sim/camera")
    async def get_sim_camera():
        payload = load_latest_frame_payload()
        return {
            "ok": True,
            "camera": payload.get("camera"),
            "has_frame": bool(payload.get("has_frame")),
        }

    @app.post("/api/sim/camera")
    async def set_sim_camera(request: Request):
        try:
            payload = await request.json()
            if not isinstance(payload, dict):
                raise ValueError("camera payload must be an object")
            result = await call_camera_mcp_tool(payload)
            sim_frame_cache.clear()
            return {"ok": True, "result": jsonable_encoder(result)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

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
                            f"event: frame\ndata: "
                            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                        )
                    elif current_ts > last_ts:
                        last_ts = current_ts
                        idle_ticks = 0
                        last_emit_ts = time.time()
                        yield (
                            f"event: frame\ndata: "
                            f"{json.dumps(payload, ensure_ascii=False)}\n\n"
                        )
                    else:
                        idle_ticks += 1
                else:
                    idle_ticks += 1

                if idle_ticks >= 100:
                    idle_ticks = 0
                    yield "event: ping\ndata: {}\n\n"

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

    @app.post("/api/sim/reset")
    async def reset_sim_environment():
        """Call MCP cleanup_simulation_tool to reset the PyBullet environment.
        Used by frontend when user starts a new conversation.
        """
        try:
            await call_mcp_tool("cleanup_simulation_tool", {})
            return {"ok": True, "message": "Simulation environment cleared"}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    @app.get("/api/sim/mjpeg")
    async def mjpeg_stream(request: Request):
        """Serve the simulation frames as an MJPEG (multipart/x-mixed-replace) stream.

        The browser renders this natively via <img src="..."/> with zero flicker,
        because each new frame atomically replaces the previous pixel buffer.
        """
        boundary = b"frame"

        async def generate():
            last_mtime = 0.0
            while True:
                if await request.is_disconnected():
                    break
                try:
                    if not sim_frame_file.exists():
                        await asyncio.sleep(0.1)
                        continue
                    mtime = sim_frame_file.stat().st_mtime
                    if mtime <= last_mtime:
                        await asyncio.sleep(0.05)
                        continue
                    last_mtime = mtime
                    with open(sim_frame_file, "rb") as f:
                        data = f.read()
                    if not data:
                        await asyncio.sleep(0.05)
                        continue
                    yield b"--" + boundary + b"\r\n"
                    yield b"Content-Type: image/png\r\n"
                    yield f"Content-Length: {len(data)}\r\n\r\n".encode("ascii")
                    yield data
                    yield b"\r\n"
                except Exception:
                    await asyncio.sleep(0.1)

        return StreamingResponse(
            generate(),
            media_type="multipart/x-mixed-replace; boundary=frame",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate, private",
                "Pragma": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
