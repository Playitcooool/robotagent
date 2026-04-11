---
name: gazebo
description: "Gazebo ROS2 physics simulator MCP tools — spawn/delete URDF/SDF models, get/set model states, camera capture, physics control. Use this skill when working with Gazebo simulation for robot manipulation, world editing, or sensor data collection."
---

# Gazebo Simulator — Tool Guide

Gazebo MCP server (`mcp/gazebo_mcp_server.py`) exposes Gazebo physics simulator via ROS2 services. All tools live under the `gazebo_simulator` MCP server and communicate over ROS2 topics/services.

**Server:** `http://localhost:8002` (Docker: `http://gazebo_mcp:8002`)

**Requires:** Running Gazebo with `gazebo_ros` packages (ROS2). Service clients connect to `/spawn_entity`, `/delete_entity`, `/get_entity_state`, `/set_entity_state`, `/pause_physics`, `/unpause_physics`, `/reset_simulation`, `/reset_world`.

---

## Tool Inventory

### 1. `initialize_ros_connection`

Initialize a fresh ROS2 node for Gazebo services/topics. Always shuts down any existing connection first — safe to call at the start of each query.

**Returns:**
- `node_name`, `message`

**When NOT to use:**
- Mid-task before pending Gazebo calls complete.
- When subsequent Gazebo tools are expected immediately (re-init will be required).

---

### 2. `spawn_model`

Spawn a URDF/SDF model entity into Gazebo.

**Parameters:**
- `model_name` (str, required): Unique entity name. Must not conflict with existing names.
- `model_xml` (str, default=""): Inline URDF/SDF XML content.
- `model_path` (str, default=""): Absolute filesystem path to URDF/SDF file.
- `position` (list[3], default=[0,0,0]): Spawn position `[x, y, z]` in meters (world frame).
- `orientation` (list[4], default=[0,0,0,1]): Quaternion `[x, y, z, w]` (world frame).
- `robot_namespace` (str, default=""): ROS2 namespace for spawned model plugins/topics.

**Priority:** If both `model_xml` and `model_path` are empty, falls back to built-in assets at `<mcp>/assets/gazebo_models/<model_name>/model.sdf`.

**Built-in models:** `unit_box`, `unit_sphere`, `unit_cylinder`. Also settable via `GAZEBO_BUILTIN_MODEL_ROOT`.

**Returns:**
- `status` ("success"/"failure"), `model_name`, `message`

**When NOT to use:**
- For moving existing models — use `set_model_state`.
- With duplicate `model_name` unless replacement is explicitly desired.

---

### 3. `list_builtin_models`

List built-in static models shipped with this MCP server.

**Returns:**
- `root` (model root directory), `models` (list of available model names)

---

### 4. `delete_model`

Delete one model/entity from Gazebo by exact name.

**Parameters:**
- `model_name` (str, required): Exact Gazebo model/entity name.

**Returns:**
- `status`, `model_name`, `message`

**When NOT to use:**
- For clearing the entire world — use `reset_world`/`reset_simulation`.
- On unknown names without first checking `list_models`.

---

### 5. `get_model_state`

Query pose and twist of one Gazebo model (single snapshot).

**Parameters:**
- `model_name` (str, required): Gazebo model/entity name.
- `reference_frame` (str, default="world"): Reference frame for returned pose/twist.

**Returns:**
- `position` [x,y,z], `orientation` [x,y,z,w], `linear_velocity` [vx,vy,vz], `angular_velocity` [wx,wy,wz]

**When NOT to use:**
- For continuous streaming — this is a single snapshot call.
- For listing all models — use `list_models`.

---

### 6. `set_model_state`

Instantaneously teleport a model to a target pose/twist (teleport-style, not physics-based).

**Parameters:**
- `model_name` (str, required): Model/entity name to move.
- `position` (list[3], required): Target `[x, y, z]` in meters.
- `orientation` (list[4], default=[0,0,0,1]): Target quaternion.
- `linear_velocity` (list[3], default=[0,0,0]): Target `[vx, vy, vz]` in m/s.
- `angular_velocity` (list[3], default=[0,0,0]): Target `[wx, wy, wz]` in rad/s.
- `reference_frame` (str, default="world"): Reference frame.

**Returns:**
- `status`, `model_name`, `message`

**When NOT to use:**
- As a physically realistic motion controller — this is instantaneous state set.
- For spawning new models — use `spawn_model`.

---

### 7. `list_models`

List all models currently reported via `/model_states`.

**Returns:**
- `model_count`, `models` (list of `{name, position}`)

**When NOT to use:**
- When full twist/orientation per model is needed — use `get_model_state` per model.
- If `/model_states` is not publishing yet (returns empty/warning).

---

### 8. `pause_simulation`

Pause Gazebo physics stepping. Use before deterministic state edits or freezing the scene.

**When NOT to use:**
- If simulation is already paused (idempotent — no error, but wasteful).

---

### 9. `unpause_simulation`

Resume Gazebo physics stepping after `pause_simulation`.

**When NOT to use:**
- If simulation is already running (idempotent).

---

### 10. `reset_simulation`

Full simulation reset: reset simulation clock AND model states (hard episode restart).

**When NOT to use:**
- When simulation time continuity is needed — use `reset_world` instead.

---

### 11. `reset_world`

World reset: reset model states but keep simulation clock (soft episode restart).

**When NOT to use:**
- When full time reset is needed — use `reset_simulation` instead.

---

### 12. `capture_camera`

Capture one camera frame from a ROS2 image topic, return as PNG base64.

**Parameters:**
- `topic` (str, default="/camera/image_raw"): ROS2 image topic (`sensor_msgs/Image`, rgb8/bgr8 preferred).
- `timeout` (float, default=5.0): Max seconds to wait for one frame before timeout.

**Returns:**
- `status` ("success"/"timeout"), `topic`, `image_base64` (PNG, on success), `format`="png"

**When NOT to use:**
- For high-rate continuous video streaming — this is single-frame polling.
- On non-image topics.

---

### 13. `cleanup_ros_connection`

Shutdown ROS2 node/executor and release all Gazebo MCP ROS resources.

**When NOT to use:**
- Mid-task before pending Gazebo calls complete.
- If subsequent Gazebo tools are expected immediately (re-init required).

---

### 14. `clear_simulation_state`

Clear cached simulation state (model states and camera frames) **without** shutting down the ROS2 connection. Use between queries to prevent cross-contamination.

**When NOT to use:**
- As a substitute for `reset_simulation`/`reset_world` (those reset Gazebo world state, this only clears the MCP server's local cache).
- Use `cleanup_ros_connection` when the session is fully done.

---

### 15. `get_simulation_info`

Get Gazebo simulation basic info.

**Returns:**
- `model_count`, `message`

**When NOT to use:**
- For per-model detailed state — use `get_model_state`/`list_models`.

---

### 16. `apply_force`

**WARNING: NOT IMPLEMENTED.** Requires `gazebo_ros_pkgs` `/gazebo/ApplyBodyWrench` service. Always returns `status: "warning"`.

**Parameters:**
- `model_name` (str): Target model.
- `force` (list[3], default=[0,0,10]): Force vector `[Fx, Fy, Fz]` in N.
- `position` (list[3], default=[0,0,0]): Application point offset from model center.

**Workaround:** Use `set_model_state` for teleportation-style force effects.

---

### 17. `move_object`

Atomic shorthand for `set_model_state` with zero velocities (simplified teleport).

**Parameters:**
- `model_name` (str, required): Model to move.
- `position` (list[3], default=[0,0,0.5]): Target `[x,y,z]`.
- `orientation` (list[4], default=[0,0,0,1]): Target quaternion.

**Returns:** Same as `set_model_state` (status, model_name, message).

**When NOT to use:**
- For continuous/non-instantaneous motion — use multiple `set_model_state` calls.

---

### 18. `create_simple_object`

Create a simple geometry (box/sphere/cylinder) using built-in SDF shapes. Simpler alternative to `spawn_model` for basic test objects.

**Parameters:**
- `name` (str, default="cube"): Object name.
- `shape` (str, default="box"): One of `box`, `sphere`, `cylinder`.
- `position` (list[3], default=[0,0,0.5]): Spawn `[x,y,z]`.
- `size` (list[3]): Size params. Box: `[x,y,z]`, Sphere: `[radius]`, Cylinder: `[radius, height]`.

**Returns:** Same structure as `spawn_model` (status, model_name, message).

**When NOT to use:**
- For complex URDF/SDF models — use `spawn_model` with full XML.

---

## Common Workflows

### Workflow A: Spawn and Manipulate a Model

```
1. initialize_ros_connection()
2. list_builtin_models()                        # see available assets
3. spawn_model(model_name="my_robot", model_path="/path/to/robot.sdf",
               position=[0,0,0])
4. get_model_state(model_name="my_robot")       # verify spawn
5. set_model_state(model_name="my_robot",
                   position=[1,0,0], linear_velocity=[0.5,0,0])
6. list_models()                               # check world state
7. cleanup_ros_connection()
```

### Workflow B: Simple Object Experiment

```
1. initialize_ros_connection()
2. create_simple_object(name="test_cube", shape="box",
                       position=[0,0,0.5], size=[0.1,0.1,0.1])
3. move_object(model_name="test_cube", position=[1,0,0.5])
4. get_model_state(model_name="test_cube")
5. delete_model(model_name="test_cube")
6. cleanup_ros_connection()
```

### Workflow C: Camera Capture

```
1. initialize_ros_connection()
2. capture_camera(topic="/camera/image_raw", timeout=5.0)  # returns PNG base64
3. cleanup_ros_connection()
```

### Workflow D: Episode Reset

```
1. initialize_ros_connection()
2. spawn_model(...) or create_simple_object(...)
3. ... experiment ...
4. reset_world()                             # keep clock, reset positions
   OR
   reset_simulation()                        # full reset (clock + positions)
5. cleanup_ros_connection()
```

---

## ROS2 Service Map

| Tool | ROS2 Service | Message Type |
|------|-------------|--------------|
| `spawn_model` | `/spawn_entity` | `gazebo_msgs/SpawnEntity` |
| `delete_model` | `/delete_entity` | `gazebo_msgs/DeleteEntity` |
| `get_model_state` | `/get_entity_state` | `gazebo_msgs/GetEntityState` |
| `set_model_state` | `/set_entity_state` | `gazebo_msgs/SetEntityState` |
| `pause_simulation` | `/pause_physics` | `std_srvs/Empty` |
| `unpause_simulation` | `/unpause_physics` | `std_srvs/Empty` |
| `reset_simulation` | `/reset_simulation` | `std_srvs/Empty` |
| `reset_world` | `/reset_world` | `std_srvs/Empty` |

## Built-in Model Assets

Located at `mcp/assets/gazebo_models/` (configurable via `GAZEBO_BUILTIN_MODEL_ROOT`):

| Model | Description |
|-------|-------------|
| `unit_box` | 1×1×1 meter box |
| `unit_sphere` | 1 meter radius sphere |
| `unit_cylinder` | 1 meter radius, 1 meter height cylinder |

These resolve `model://` URIs within SDF files.

---

## Error Handling

- **Service unavailable:** `call_service_sync` raises `RuntimeError` if the service doesn't appear within 5s. Verify Gazebo and `gazebo_ros` are running.
- **Validation errors** (NaN, Inf, out-of-range vectors): Return `status: "error"` before any ROS call.
- **Model not found:** `get_model_state`/`delete_model` return `status: "failure"` from the ROS service.
- **`/model_states` not publishing:** `list_models` returns `status: "warning"` with empty list. Ensure `gazebo_ros` plugin is loaded.
- **Camera timeout:** `capture_camera` returns `status: "timeout"` after the specified timeout. Verify the image topic is publishing.
- **`apply_force`:** Always returns `status: "warning"` — not implemented. Use `set_model_state` as a workaround.

---

## State Management

- Each tool call that uses `ensure_ros()` creates or reuses a cached ROS node. The node persists across calls within a session.
- `clear_simulation_state()` clears the MCP server's local cache of model states and camera frames — use between independent queries to prevent cross-contamination.
- `cleanup_ros_connection()` fully tears down the ROS node — needed when the session ends.
- Vector validation (`_validate_vector`) is performed client-side before any ROS call: checks for NaN/Inf, size, sign, and range.
