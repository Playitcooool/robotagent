---
name: pybullet
description: "PyBullet physics simulator MCP tools — headless/DIRECT mode, realtime frame streaming, KUKA arm and cube manipulation. Use this skill when working with PyBullet simulation for motion control, pick-and-place, or physics experiments."
---

# PyBullet Simulator — Tool Guide

PyBullet MCP server (`mcp/mcp_server.py`) exposes a headless/direct-mode physics simulator with realtime frame streaming. All tools live under the `pybullet_simulator` MCP server.

**Server:** `http://localhost:8000` (Docker: `http://pybullet_mcp:8000`)

---

## Tool Inventory

### 1. `initialize_simulation`

Initialize a fresh PyBullet world. Always disconnects any existing connection first — safe to call at the start of each query.

**Parameters:**
- `gui` (bool, default=False): Open PyBullet GUI window. `False` is recommended for server/agent usage (DIRECT mode is faster and headless).

**Returns:**
- `status`, `physicsClientId`, `asset_status` (map of available bundled assets), `message`

**When NOT to use:**
- Do not call mid-episode unless you intentionally want to restart the world.
- Do not use `gui=True` in headless/server environments.

**Streams:** Publishes a snapshot frame to `~/.sim_stream/latest.png` on init.

---

### 2. `check_static_assets`

Check if required PyBullet bundled assets (`plane.urdf`, `cube_small.urdf`, `kuka_iiwa/model.urdf`) are available via `pybullet_data`.

**Returns:**
- `data_path`, `asset_status` (per-asset bool), `missing_assets`

**When NOT to use:**
- After `initialize_simulation` succeeds, assets are guaranteed available.

---

### 3. `push_cube_step`

Run a deterministic cube-pushing sequence with realtime frame streaming. Creates a cube and moves it incrementally along a push vector.

**Parameters:**
- `start_position` (list[3], default=[0,0,0.02]): Cube start `[x, y, z]` in meters.
- `push_vector` (list[3], default=[0.2,0,0]): Translation vector `[dx, dy, dz]` applied over the full episode.
- `steps` (int, default=120): Number of incremental steps. Higher = smoother motion.

**Returns:**
- `final_position`, `object_id`, `stream_id`, `stream_meta_path`

**When NOT to use:**
- For articulated robot manipulation — this is a cube baseline.
- With non-positive steps.

---

### 4. `grab_and_place_step`

Simplified pick-and-place episode: lift object to a transport pose, then move/place at target.

**Parameters:**
- `start_position` (list[3], default=[0.2,0,0.02]): Object start `[x,y,z]`.
- `target_position` (list[3], default=[0.4,0.4,0.02]): Target `[x,y,z]`.
- `steps` (int, default=120): Placement phase step count.

**Returns:**
- `final_position`, `object_id`, `stream_id`, `stream_meta_path`

**When NOT to use:**
- When precise grasp/contact planning is needed — this is a simplified teleport-style routine.
- With non-positive steps.

---

### 5. `path_planning`

Linear path interpolation for KUKA IIWA arm base motion.

**Parameters:**
- `start_position` (list[3], default=[0.2,0,0.02]): Robot base start.
- `target_position` (list[3], default=[0.4,0.4,0.02]): Robot base target.
- `steps` (int, default=240): Linear interpolation steps.

**Returns:**
- `final_position`, `object_id`, `stream_id`, `stream_meta_path`

**When NOT to use:**
- When obstacle avoidance or IK-level realism is needed — this is linear interpolation only.
- With non-positive steps.

---

### 6. `adjust_physics`

Create a test cube and set its friction/restitution for surface interaction experiments.

**Parameters:**
- `friction` (float, default=0.5): Lateral friction coefficient, range [0.0, 10.0].
- `restitution` (float, default=0.9): Bounciness, range [0.0, 1.0].

**Returns:**
- `object_id`, `message`

**When NOT to use:**
- As a motion-control action — it only changes dynamics parameters.
- Expecting global scene-wide physics update (currently per-cube only).

---

### 7. `multi_object_grab_and_place`

Batch-move multiple cubes to a single target location.

**Parameters:**
- `object_positions` (list[list[3]]): List of start positions `[[x,y,z], ...]`.
- `target_position` (list[3]): Shared target `[x,y,z]`.

**Returns:**
- `object_ids`, `target_position`, `status` (warning if any failures)

**When NOT to use:**
- For sequential collision-aware pick-place.
- When per-object trajectories are needed.

---

### 8. `simulate_vision_sensor`

Capture one camera image from the current scene.

**Parameters:**
- `width` (int, default=640): Image width in pixels (max 4096).
- `height` (int, default=480): Image height in pixels (max 4096).
- `view_matrix` (list[16]): Flattened 4x4 PyBullet view matrix.
- `projection_matrix` (list[16]): Flattened 4x4 projection matrix.

**Returns:**
- `image` (RGB array), `message`

**When NOT to use:**
- When only object state/poses is needed — use `check_simulation_state`.
- Expecting compressed/serialized output for direct JSON transport (use `_publish_snapshot` for streaming).

---

### 9. `cleanup_simulation_tool`

Tear down the active simulation connection and clear local state.

**When NOT to use:**
- Mid-episode unless intentionally ending the current world.
- Before state-inspection tools that need an active simulation.

---

### 10. `check_simulation_state`

Read-only inspection: return per-body base positions for all objects.

**Returns:**
- `state_data` (dict: `body_id -> position`), `num_objects`, `message`

**When NOT to use:**
- As a motion action.
- For detailed joint-level kinematics (returns only base positions).

Also publishes a snapshot frame so UI can refresh.

---

### 11. `reset_simulation`

Reset the simulation world.

**Parameters:**
- `keep_objects` (bool, default=True): True = reset positions/velocities but keep bodies; False = remove all non-plane bodies.

**Returns:**
- `removed_ids`, `kept_ids`, `message`

---

### 12. `pause_simulation`

**WARNING: PyBullet DIRECT mode does not support true pause.** Returns a warning status. Use `step_simulation` with `steps=1` to manually control timing.

---

### 13. `unpause_simulation`

**WARNING: PyBullet DIRECT mode does not support true unpause.** Returns a warning status. No action taken.

---

### 14. `get_object_state`

Get detailed state for a specific body.

**Parameters:**
- `object_id` (int): Body ID from `check_simulation_state`.

**Returns:**
- `position` [x,y,z], `orientation` [x,y,z,w], `linear_velocity` [x,y,z], `angular_velocity` [x,y,z]

**When NOT to use:**
- For bodies that have been removed.

---

### 15. `set_object_position`

Atomically teleport an object to a target pose (no physics, instantaneous).

**Parameters:**
- `object_id` (int): Body ID.
- `position` (list[3], default=[0.5,0,0.5]): Target `[x,y,z]`.
- `orientation` (list[4], default=[0,0,0,1]): Target quaternion `[x,y,z,w]`.

**Returns:**
- `position`, `orientation` after set.

**When NOT to use:**
- For physically realistic motion (use multiple `step_simulation` calls instead).

---

### 16. `step_simulation`

Execute a fixed number of physics steps manually.

**Parameters:**
- `steps` (int, default=1): Step count. Clamped to [1, 10000] per call.

**Returns:**
- `steps` (actual steps executed), `message`

**When NOT to use:**
- Without an active simulation (call `initialize_simulation` first).

---

### 17. `create_object`

Create a single rigid body (cube/sphere/cylinder) with custom mass and color.

**Parameters:**
- `object_type` (str, default="cube"): One of `cube`, `sphere`, `cylinder`.
- `position` (list[3], default=[0,0,0.5]): Spawn position `[x,y,z]`.
- `size` (list[3]): Dimensions. Cube: `[x,y,z]`, Sphere: `[radius]`, Cylinder: `[radius, height]`.
- `mass` (float, default=1.0): Mass in kg.
- `color` (list[4], default=[1,0,0,1]): RGBA color.

**Returns:**
- `object_id`, `object_type`, `position`, `message`

---

### 18. `delete_object`

Remove a body from the simulation.

**Parameters:**
- `object_id` (int): Body ID to remove.

**Returns:**
- `deleted_id`, `message`

**When NOT to use:**
- On already-removed bodies (check `check_simulation_state` first).

---

### 19. `get_simulation_info`

Query simulation engine parameters.

**Returns:**
- `timestep`, `num_bodies`, `gravity` [x,y,z], `message`

---

### 20. `set_gravity`

Set global gravity vector.

**Parameters:**
- `gravity` (list[3], default=[0,0,-9.8]): Gravity vector `[x,y,z]`.

**Returns:**
- `gravity` (confirmed set value), `message`

**Use cases:** Zero-g experiments, Moon/Mars gravity simulation.

---

## Common Workflows

### Workflow A: Cube Motion Experiment

```
1. initialize_simulation(gui=False)
2. create_object(object_type="cube", position=[0,0,0.02], size=[0.05,0.05,0.05])
3. get_object_state(object_id=<id>)       # verify initial state
4. set_object_position(object_id=<id>, position=[0.5,0,0.02])
5. step_simulation(steps=240)             # let physics settle
6. check_simulation_state()              # inspect result
7. cleanup_simulation_tool()
```

### Workflow B: Pick-and-Place

```
1. initialize_simulation(gui=False)
2. grab_and_place_step(start_position=[0.2,0,0.02], target_position=[0.4,0.4,0.02], steps=120)
3. get_object_state(object_id=<returned_id>)
4. cleanup_simulation_tool()
```

### Workflow C: Multi-Object Scene Setup

```
1. initialize_simulation(gui=False)
2. multi_object_grab_and_place(
     object_positions=[[0.2,0,0.02],[0.4,0.4,0.02]],
     target_position=[0.6,0.6,0.02]
   )
3. check_simulation_state()
4. cleanup_simulation_tool()
```

---

## Error Handling

- **Validation errors** (NaN, Inf, out-of-range): Return `status: "error"` with descriptive message. Check vector bounds before calling.
- **Body not found**: `get_object_state` / `delete_object` return `status: "warning"` if the body ID is invalid.
- **Simulation not initialized**: Most tools auto-initialize via `ensure_simulation()` context manager.
- **`gui=True` in headless environment**: `initialize_simulation` will fail — only use GUI mode when a display is available.
- **`pybullet_data` assets missing**: `check_static_assets` returns `status: "warning"` with `missing_assets` list.

---

## Shared State

PyBullet uses a **single global client ID** (`simulation_instance`). All API calls through `with_simulation()` context manager ensure the correct client is targeted. If multiple agents call `initialize_simulation` concurrently, each creates a fresh environment (disconnecting the previous).

Realtime frames are written to `PYBULLET_STREAM_DIR` (default: `mcp/.sim_stream/`):
- `latest.json` — frame metadata (run_id, step, done, timestamp)
- `latest.png` — most recent RGB frame

Frame files persist until `cleanup_simulation_tool` is called.
