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

### 18. `load_urdf`

Load a robot or articulated URDF model.

**Parameters:**
- `urdf_path` (str): Built-in examples include `kuka_iiwa/model.urdf` and `franka_panda/panda.urdf`. Use KUKA for the current fixed-constraint grasp demo; Panda needs gripper-specific open/close tooling for realistic grasping.
- `position` (list[3], default=[0,0,0]): Base position.
- `orientation` (list[4], default=[0,0,0,1]): Base quaternion.
- `use_fixed_base` (bool, default=True): Keep robot base fixed.

**Returns:**
- `object_id`, `urdf_path`, `position`, `num_joints`

---

### 19. `move_end_effector`

Move a robot tool link to a world-space target with inverse kinematics. Auto-detects common tool links such as KUKA `lbr_iiwa_link_7` and Panda `panda_hand` / `panda_link8`; Panda finger links are avoided by default.

**Parameters:**
- `object_id` (int): Robot body ID from `load_urdf`.
- `target_position` (list[3]): Desired tool position `[x,y,z]`.
- `end_effector_index` (int, default=-1): Explicit link override; `-1` uses auto-detection.
- `max_force` (float, default=500): Motor force.

**Returns:**
- `end_effector_link`, `end_effector_position`, `joint_positions`

**Workflow:** Call `step_simulation(steps=200+)` after this tool so the arm physically moves to the IK target.

---

### 20. `grasp_object`

Attach a nearby object to the selected robot end-effector with a fixed constraint. The tool computes the real object pose relative to the tool link at grasp time, so the object follows without the old hard-coded offset.

**Parameters:**
- `robot_id` (int): Robot body ID.
- `object_id` (int): Object body ID to grasp.
- `end_effector_index` (int, default=-1): Explicit link override; `-1` uses the shared auto-detection used by `move_end_effector`.
- `max_grasp_distance` (float, default=0.08): Maximum allowed distance from tool to object AABB/center. Keep this tight so objects do not appear to attach from far away.
- `snap_to_tool` (bool, default=False): Keep the object in place by default so the robot visibly reaches it. Set true only for explicit snap/quick-demo behavior.

**Returns:**
- `constraint_id`, `end_effector_link`, `end_effector_position`, `object_position`, `grasp_distance`, `snapped`

**Recommended workflow:** move above the object first, descend to a non-colliding grasp height, then grasp:
`move_end_effector(pregrasp) -> step_simulation -> move_end_effector(grasp_height) -> step_simulation -> grasp_object -> move_end_effector(place) -> step_simulation -> release_object`.

---

### 21. `release_object`

Remove the active fixed grasp constraint for an object.

**Parameters:**
- `object_id` (int): Object body ID to release.

---

### 22. `delete_object`

Remove a body from the simulation.

**Parameters:**
- `object_id` (int): Body ID to remove.

**Returns:**
- `deleted_id`, `message`

**When NOT to use:**
- On already-removed bodies (check `check_simulation_state` first).

---

### 23. `get_simulation_info`

Query simulation engine parameters.

**Returns:**
- `timestep`, `num_bodies`, `gravity` [x,y,z], `message`

---

### 24. `set_gravity`

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
2. load_urdf(urdf_path="kuka_iiwa/model.urdf", position=[0,0,0], use_fixed_base=True)
3. create_object(object_type="cube", position=[0.4,0,0.05], size=[0.05,0.05,0.05], mass=0.1)
4. move_end_effector(object_id=<robot_id>, target_position=[0.4,0,0.25])
5. step_simulation(steps=300)
6. move_end_effector(object_id=<robot_id>, target_position=[0.4,0,0.12])
7. step_simulation(steps=300)
8. grasp_object(robot_id=<robot_id>, object_id=<cube_id>)
9. move_end_effector(object_id=<robot_id>, target_position=[0.0,0.4,0.3])
10. step_simulation(steps=300)
11. release_object(object_id=<cube_id>)
12. cleanup_simulation_tool()
```

The cube target should be near the object center or just above it. `grasp_object`
returns `grasp_distance` plus `snapped` for diagnosis. By default it does not
teleport the object; use `snap_to_tool=true` only when snap-to-gripper behavior
is explicitly desired. If the object moves before `grasp_object`, the arm path
or target height is colliding with the object; use a higher pregrasp point and
keep the grasp target slightly above the object.

For fixed-constraint pick-and-place, prefer `kuka_iiwa/model.urdf`. The bundled
Panda model exposes a hand/finger structure, but these tools do not yet open and
close the gripper, so Panda IK may stop too far from the cube for a visually
credible grasp.

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
