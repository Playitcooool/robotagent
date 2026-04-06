#!/usr/bin/env python3
"""
charts/ 生成脚本 —— 论文所需各类统计图表，全部单独输出，中文标签
每个函数生成一张独立图片，直接读取数据，无需重新运行实验。
"""

from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "charts" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── 样式 ──────────────────────────────────────────────
plt.rcParams.update({
    "font.sans-serif": ["PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei", "Arial Unicode MS"],
    "axes.unicode_minus": False,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "grid.color": "#d8d8d8",
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})


# ═══════════════════════════════════════════════════════════
# 图1a：RAG 知识库——原始文档数
# ═══════════════════════════════════════════════════════════

def fig_rag_doc_counts():
    sources = {
        "ROS 2 Humble\n官方文档": {"docs": 6, "color": "#1565c0"},
        "PyBullet\n官方论坛": {"docs": 116, "color": "#e53935"},
        "ManiSkill\n官方文档": {"docs": 4, "color": "#43a047"},
        "Gazebo\n官方文档": {"docs": 4, "color": "#fb8c00"},
    }
    names = list(sources.keys())
    counts = [s["docs"] for s in sources.values()]
    colors = [s["color"] for s in sources.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.8, width=0.55)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
                str(val), ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_title("RAG 知识库——原始文档数", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("文档数量（份）", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.22)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rag_doc_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: rag_doc_counts.png")


# ═══════════════════════════════════════════════════════════
# 图1b：RAG 知识库——Chunk 数
# ═══════════════════════════════════════════════════════════

def fig_rag_chunk_counts():
    sources = {
        "ROS 2 Humble\n官方文档": {"chunks": 374, "color": "#1565c0"},
        "PyBullet\n官方论坛": {"chunks": 47, "color": "#e53935"},
        "ManiSkill\n官方文档": {"chunks": 31, "color": "#43a047"},
        "Gazebo\n官方文档": {"chunks": 3, "color": "#fb8c00"},
    }
    names = list(sources.keys())
    counts = [s["chunks"] for s in sources.values()]
    colors = [s["color"] for s in sources.values()]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(names, counts, color=colors, edgecolor="white", linewidth=1.8, width=0.55)
    for bar, val in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=14, fontweight="bold")
    ax.set_title("RAG 知识库——切分 Chunk 数", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("Chunk 数量（块）", fontsize=12)
    ax.set_ylim(0, max(counts) * 1.22)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rag_chunk_counts.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: rag_chunk_counts.png")


# ═══════════════════════════════════════════════════════════
# 图1c：RAG Chunk 来源占比饼图
# ═══════════════════════════════════════════════════════════

def fig_rag_chunk_pie():
    labels = ["ROS 2 Humble", "PyBullet", "ManiSkill", "Gazebo"]
    chunks = [374, 47, 31, 3]
    colors = ["#1565c0", "#e53935", "#43a047", "#fb8c00"]
    explode = [0.06, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        chunks, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2.5},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    for text in texts:
        text.set_fontsize(12)

    ax.set_title("RAG Chunk 来源占比", fontsize=14, fontweight="bold", pad=15)
    ax.legend(
        wedges, [f"{l}（{c}块）" for l, c in zip(labels, chunks)],
        loc="lower right", fontsize=11, framealpha=0.9
    )
    fig.tight_layout()
    fig.savefig(OUT_DIR / "rag_chunk_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: rag_chunk_pie.png")


# ═══════════════════════════════════════════════════════════
# 图2：MCP 工具按功能分类统计（PyBullet vs Gazebo 双柱）
# ═══════════════════════════════════════════════════════════

def fig_mcp_tools():
    categories = {
        "环境管理": ["initialize_simulation", "initialize_ros_connection", "check_static_assets",
                     "cleanup_simulation_tool", "cleanup_ros_connection", "clear_simulation_state",
                     "reset_simulation", "reset_world", "pause_simulation", "unpause_simulation"],
        "物体操作": ["create_object", "create_simple_object", "delete_object", "delete_model",
                     "set_object_position", "set_model_state", "move_object"],
        "任务执行": ["push_cube_step", "grab_and_place_step", "multi_object_grab_and_place",
                     "path_planning", "apply_force"],
        "状态感知": ["get_object_state", "get_model_state", "check_simulation_state",
                     "list_models", "list_builtin_models", "get_simulation_info"],
        "视觉传感": ["simulate_vision_sensor", "capture_camera"],
        "物理调节": ["adjust_physics", "set_gravity"],
    }

    pybullet_tools_raw = [
        "initialize_simulation", "check_static_assets", "push_cube_step",
        "grab_and_place_step", "path_planning", "adjust_physics",
        "multi_object_grab_and_place", "simulate_vision_sensor",
        "cleanup_simulation_tool", "check_simulation_state",
        "reset_simulation", "pause_simulation", "unpause_simulation",
        "get_object_state", "set_object_position", "step_simulation",
        "create_object", "delete_object", "get_simulation_info", "set_gravity",
    ]
    gazebo_tools_raw = [
        "initialize_ros_connection", "spawn_model", "list_builtin_models",
        "delete_model", "get_model_state", "set_model_state", "list_models",
        "pause_simulation", "unpause_simulation", "reset_simulation",
        "reset_world", "capture_camera", "cleanup_ros_connection",
        "clear_simulation_state", "get_simulation_info", "apply_force",
        "move_object", "create_simple_object",
    ]

    def count_cat(tool_list, cat_tools):
        return sum(1 for t in tool_list if any(ct in t for ct in cat_tools))

    cat_names = list(categories.keys())
    pb_vals = [count_cat(pybullet_tools_raw, categories[c]) for c in cat_names]
    gaz_vals = [count_cat(gazebo_tools_raw, categories[c]) for c in cat_names]

    x = np.arange(len(cat_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, pb_vals, width, label="PyBullet", color="#e53935", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width/2, gaz_vals, width, label="Gazebo", color="#1565c0", edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars1, pb_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#b71c1c")
    for bar, val in zip(bars2, gaz_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontsize=11, fontweight="bold", color="#0d47a1")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=11)
    ax.set_ylabel("工具数量（个）", fontsize=12)
    ax.set_ylim(0, max(max(pb_vals), max(gaz_vals)) + 3)
    ax.legend(loc="upper right", fontsize=12)
    ax.set_title(f"MCP 工具按功能分类统计（PyBullet {len(pybullet_tools_raw)} 个 / Gazebo {len(gazebo_tools_raw)} 个）",
                 fontsize=14, fontweight="bold", pad=12)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "mcp_tools_category.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: mcp_tools_category.png")


# ═══════════════════════════════════════════════════════════
# 图3：技术栈信息表
# ═══════════════════════════════════════════════════════════

def fig_tech_stack():
    tech_stack = [
        ("LLM", "Qwen3.5-9B（本地量化）"),
        ("框架", "LangChain + ReAct"),
        ("协议", "MCP（Model Context Protocol）"),
        ("向量库", "Qdrant（本地部署）"),
        ("缓存/消息", "Redis（上下文共享 + Pub/Sub）"),
        ("前端", "Vue3 + Canvas 实时渲染"),
        ("后端", "FastAPI（REST API + SSE）"),
        ("容器化", "Docker Compose"),
        ("模型推理", "LM Studio（OpenAI-compatible API）"),
    ]

    fig, ax = plt.subplots(figsize=(9, 7))
    ax.axis("off")

    cell_data = [[t, d] for t, d in tech_stack]
    tbl = ax.table(
        cellText=cell_data, colLabels=["技术", "选型"],
        loc="center", cellLoc="left",
        bbox=[0.0, 0.0, 1.0, 1.0],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.55)
    # 表头
    for j in range(2):
        tbl[(0, j)].set_facecolor("#1565c0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold", fontsize=12)
    # 奇偶行
    for i in range(1, len(cell_data) + 1):
        fc = "#e3f2fd" if i % 2 == 1 else "white"
        tbl[(i, 0)].set_facecolor(fc)
        tbl[(i, 0)].set_text_props(fontweight="bold")
        tbl[(i, 1)].set_facecolor(fc)

    ax.set_title("技术栈", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "tech_stack.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: tech_stack.png")


# ═══════════════════════════════════════════════════════════
# 图4：三智能体角色
# ═══════════════════════════════════════════════════════════

def fig_agent_roles():
    agents = [
        ("主智能体", "任务理解、规划协调\nReAct 推理、工具调度决策", "#1565c0", "#e3f2fd"),
        ("仿真智能体", "专职执行仿真操作\nPyBullet / Gazebo API 调用", "#e53935", "#ffebee"),
        ("分析智能体", "结果评估、误差分析\n质量评分、策略建议生成", "#2e7d32", "#e8f5e9"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    for i, (name, desc, color, fc) in enumerate(agents):
        y = 6.5 - i * 2.2
        bbox = FancyBboxPatch((0.5, y - 0.9), 9, 1.8,
                               boxstyle="round,pad=0.15", facecolor=fc,
                               edgecolor=color, linewidth=2.5)
        ax.add_patch(bbox)
        ax.text(5, y + 0.4, name, ha="center", va="center", fontsize=15,
                fontweight="bold", color=color)
        ax.text(5, y - 0.2, desc, ha="center", va="center", fontsize=11,
                color="#333", style="italic")

    ax.set_title("三智能体角色分工", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "agent_roles.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: agent_roles.png")


# ═══════════════════════════════════════════════════════════
# 图5：核心特性卡片
# ═══════════════════════════════════════════════════════════

def fig_core_features():
    features = [
        ("Agentic RAG", "检索作为可按需调用的工具\n多路并发召回（向量库 + 互联网 + 学术库）", "#fb8c00"),
        ("MCP 协议", "仿真工具标准化接入\n热插拔工具集，新工具无需改核心代码", "#8e24aa"),
        ("经验回放", "轨迹收集 → 筛选 → few-shot 注入\n免训练策略优化", "#00838f"),
        ("实时可视化", "SSE 推送 + Canvas 帧渲染\n任务执行过程全程可观测", "#ad1457"),
        ("状态外置化", "Redis 共享上下文\n断电可恢复、多实例并行", "#4e342e"),
    ]

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 8)
    ax.axis("off")

    for i, (name, desc, color) in enumerate(features):
        y = 7.0 - i * 1.4
        bbox = FancyBboxPatch((0.3, y - 0.6), 9.4, 1.2,
                               boxstyle="round,pad=0.12", facecolor=color, alpha=0.12,
                               edgecolor=color, linewidth=2)
        ax.add_patch(bbox)
        ax.text(0.7, y + 0.15, name, ha="left", va="center", fontsize=13,
                fontweight="bold", color=color)
        ax.text(0.7, y - 0.3, desc, ha="left", va="center", fontsize=10, color="#333")

    ax.set_title("系统核心特性", fontsize=15, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "core_features.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: core_features.png")


# ═══════════════════════════════════════════════════════════
# 图6：ReAct 推理循环
# ═══════════════════════════════════════════════════════════

def fig_react_loop():
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    nodes = [
        (5.5, 6.8, "用户指令", "#e3f2fd", "#1565c0"),
        (2.5, 4.8, "Reason\n推理", "#fff3e0", "#e65100"),
        (5.5, 4.8, "Act\n工具调用", "#e8f5e9", "#2e7d32"),
        (8.5, 4.8, "Observe\n结果观察", "#fce4ec", "#ad1457"),
        (5.5, 2.8, "上下文\n更新", "#f3e5f5", "#6a1b9a"),
        (5.5, 1.0, "结束 / 输出回复", "#ffecb3", "#f57f17"),
    ]

    arrows = [
        (5.5, 6.4, 2.5, 5.2),
        (2.5, 4.4, 5.5, 5.2),
        (5.5, 4.4, 8.5, 5.2),
        (8.5, 4.4, 8.5, 3.2),
        (8.5, 3.2, 2.5, 5.2),
        (2.5, 3.2, 5.5, 3.2),
        (5.5, 2.4, 5.5, 1.4),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#888", lw=1.8))

    # 循环标注
    ax.annotate("循环", xy=(2.5, 4.4), xytext=(1.3, 3.8),
                fontsize=9, color="#e65100", style="italic",
                arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.5,
                                connectionstyle="arc3,rad=0.4"))

    for x, y, label, fc, ec in nodes:
        bbox = FancyBboxPatch((x-0.9, y-0.55), 1.8, 1.1,
                                boxstyle="round,pad=0.1", facecolor=fc,
                                edgecolor=ec, linewidth=2.2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", fontsize=11.5,
                fontweight="bold", color=ec)

    ax.text(5.5, 0.1,
             "Reason（推理下一步动作） → Act（调用 MCP 工具） → Observe（获取仿真状态） → Context 更新 → 循环",
             ha="center", va="center", fontsize=9, color="#555", style="italic",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#f5f5f5", edgecolor="#ccc"))

    ax.set_title("ReAct 推理-行动-观察循环", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.savefig(OUT_DIR / "react_loop.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: react_loop.png")


# ═══════════════════════════════════════════════════════════
# 图7：三智能体协同架构
# ═══════════════════════════════════════════════════════════

def fig_multi_agent_arch():
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def draw_agent(cx, cy, name, sub, color, w=2.4, h=1.4):
        bbox = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                boxstyle="round,pad=0.12", facecolor=color, alpha=0.12,
                                edgecolor=color, linewidth=2.5)
        ax.add_patch(bbox)
        ax.text(cx, cy + 0.2, name, ha="center", va="center", fontsize=13,
                fontweight="bold", color=color)
        ax.text(cx, cy - 0.35, sub, ha="center", va="center", fontsize=9,
                color="#333", style="italic")

    draw_agent(2.2, 5.8, "主智能体", "任务理解 / 规划协调", "#1565c0")
    draw_agent(6, 5.8, "仿真智能体", "PyBullet / Gazebo 执行", "#e53935")
    draw_agent(9.8, 5.8, "分析智能体", "质量评估 / 误差分析", "#2e7d32")

    # Redis
    redis_box = FancyBboxPatch((4.4, 3.2), 3.2, 1.3,
                                 boxstyle="round,pad=0.1", facecolor="#90a4ae", alpha=0.2,
                                 edgecolor="#546e7a", linewidth=2, linestyle="--")
    ax.add_patch(redis_box)
    ax.text(6, 4.0, "Redis", ha="center", va="center", fontsize=13,
            fontweight="bold", color="#37474f")
    ax.text(6, 3.45, "上下文共享 / Pub-Sub", ha="center", va="center",
            fontsize=9, color="#555")

    # 用户
    user_box = FancyBboxPatch((0, 5.3), 1.2, 0.9,
                               boxstyle="round,pad=0.08", facecolor="#1565c0", alpha=0.15,
                               edgecolor="#1565c0", linewidth=1.8)
    ax.add_patch(user_box)
    ax.text(0.6, 5.75, "用户", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#1565c0")

    # 箭头
    ax.annotate("", xy=(0.95, 5.75), xytext=(1.2, 5.75),
               arrowprops=dict(arrowstyle="->", color="#1565c0", lw=2))
    ax.text(1.05, 6.0, "指令", fontsize=9, color="#1565c0")

    ax.annotate("", xy=(3.5, 5.8), xytext=(4.7, 5.8),
               arrowprops=dict(arrowstyle="<->", color="#546e7a", lw=1.8))
    ax.text(4.1, 6.1, "MCP", fontsize=9, color="#546e7a")

    ax.annotate("", xy=(7.3, 5.8), xytext=(8.6, 5.8),
               arrowprops=dict(arrowstyle="<->", color="#546e7a", lw=1.8))
    ax.text(7.95, 6.1, "评估", fontsize=9, color="#546e7a")

    ax.annotate("", xy=(6, 4.5), xytext=(4.0, 5.1),
               arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.5,
                               connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(6, 3.5), xytext=(6, 5.1),
               arrowprops=dict(arrowstyle="<->", color="#78909c", lw=1.5))
    ax.annotate("", xy=(4.4, 4.5), xytext=(8.6, 5.1),
               arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.5,
                               connectionstyle="arc3,rad=-0.2"))

    # MCP
    ax.annotate("", xy=(6, 2.8), xytext=(6, 3.15),
               arrowprops=dict(arrowstyle="->", color="#90a4ae", lw=1.5))
    ax.text(6, 2.4, "MCP 协议", ha="center", va="center", fontsize=11,
            fontweight="bold", color="#546e7a")
    ax.text(6, 1.95, "PyBullet 仿真引擎  |  Gazebo 仿真引擎",
            ha="center", va="center", fontsize=9, color="#777")

    ax.set_title("三智能体协同架构", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "multi_agent_arch.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: multi_agent_arch.png")


# ═══════════════════════════════════════════════════════════
# 图8：经验回放三阶段
# ═══════════════════════════════════════════════════════════

def fig_experience_replay():
    stages = [
        (2, 3.5, "1. 轨迹收集", ["自动记录每次任务执行", "prompt / response", "tool_calls 历史"],
         "#1565c0", "#e3f2fd"),
        (6, 3.5, "2. 经验筛选", ["按智能体类型过滤", "分析智能体质量评分", "Jaccard 相似度去重"],
         "#e65100", "#fff3e0"),
        (10, 3.5, "3. Few-Shot 注入", ["TopK 筛选高质量经验", "格式化 few-shot 示例", "注入新任务提示词"],
         "#2e7d32", "#e8f5e9"),
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5)
    ax.axis("off")

    for cx, cy, title, lines, color, fc in stages:
        bbox = FancyBboxPatch((cx - 1.6, cy - 1.4), 3.2, 2.8,
                                boxstyle="round,pad=0.18", facecolor=fc,
                                edgecolor=color, linewidth=2.5)
        ax.add_patch(bbox)
        ax.text(cx, cy + 1.0, title, ha="center", va="center", fontsize=13,
                fontweight="bold", color=color)
        for i, line in enumerate(lines):
            ax.text(cx, cy + 0.1 - i * 0.7, line, ha="center", va="center",
                    fontsize=10.5, color="#222")

    # 箭头
    ax.annotate("", xy=(4.45, 3.5), xytext=(3.65, 3.5),
               arrowprops=dict(arrowstyle="->", color="#888", lw=2.5))
    ax.annotate("", xy=(8.45, 3.5), xytext=(7.65, 3.5),
               arrowprops=dict(arrowstyle="->", color="#888", lw=2.5))

    ax.text(7, 1.1,
             "Score(e|q) = α · cos_sim(v_q, v_e) + β · Quality(e)        注入经验: E_inject = TopK(Score, k=3)",
             ha="center", va="center", fontsize=10, color="#444",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#fafafa", edgecolor="#ccc"))

    ax.set_title("经验回放机制：收集 → 筛选 → 注入", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout(rect=[0, 0.1, 1, 1])
    fig.savefig(OUT_DIR / "experience_replay_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: experience_replay_flow.png")


# ═══════════════════════════════════════════════════════════
# 图9：系统端到端数据流
# ═══════════════════════════════════════════════════════════

def fig_data_flow():
    nodes = [
        (1.2, 3.2, "用户\n自然语言指令", "#1565c0", "#e3f2fd"),
        (3.5, 3.2, "Vue3 前端\n任务创建 + 可视化", "#6a1b9a", "#f3e5f5"),
        (5.8, 3.2, "FastAPI 后端\nREST API + SSE", "#00838f", "#e0f7fa"),
        (8.1, 3.2, "多智能体层\n主/仿真/分析智能体", "#e65100", "#fff3e0"),
        (10.4, 3.2, "MCP 协议\n工具网关", "#2e7d32", "#e8f5e9"),
        (12.2, 3.2, "仿真引擎\nPyBullet/Gazebo", "#c62828", "#ffebee"),
    ]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6)
    ax.axis("off")

    for x, y, label, ec, fc in nodes:
        bbox = FancyBboxPatch((x - 1.05, y - 0.75), 2.1, 1.5,
                                boxstyle="round,pad=0.1", facecolor=fc,
                                edgecolor=ec, linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", fontsize=9.5,
                fontweight="bold", color=ec)

    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + 1.05
        x2 = nodes[i+1][0] - 1.05
        ax.annotate("", xy=(x2, nodes[i][1]), xytext=(x1, nodes[i][1]),
                   arrowprops=dict(arrowstyle="->", color="#888", lw=2))

    ax.text(4.65, 4.1, "Redis 上下文共享", ha="center", va="center",
            fontsize=9, color="#546e7a", style="italic")
    ax.text(7.0, 4.1, "Qdrant 向量检索", ha="center", va="center",
            fontsize=9, color="#546e7a", style="italic")

    # 反馈回路
    ax.annotate("", xy=(5.8, 1.1), xytext=(10.4, 1.1),
               arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.8,
                               connectionstyle="arc3,rad=0"))
    ax.annotate("", xy=(3.5, 2.45), xytext=(5.8, 2.45),
               arrowprops=dict(arrowstyle="<-", color="#c62828", lw=1.8,
                               connectionstyle="arc3,rad=0"))
    ax.text(7.1, 1.25, "SSE 仿真帧流 / 分析评估结果", ha="center", va="center",
            fontsize=9, color="#c62828", style="italic")

    ax.set_title("系统端到端数据流", fontsize=14, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "data_flow.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: data_flow.png")


# ═══════════════════════════════════════════════════════════
# 图10：各智能体工具集
# ═══════════════════════════════════════════════════════════

def fig_main_agent_tools():
    tools = [
        "ReAct 推理（自有能力）",
        "任务分解与规划（自有能力）",
        "search（按需调用外部检索工具）",
        "MCP 工具调度（通过仿真智能体间接调用）",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(tools) + 0.8)
    ax.axis("off")

    bbox = FancyBboxPatch((0.01, 0.05), 0.98, len(tools) + 0.55,
                           boxstyle="round,pad=0.08", facecolor="#e3f2fd",
                           edgecolor="#1565c0", linewidth=2, transform=ax.transAxes)
    ax.add_patch(bbox)
    ax.text(0.5, len(tools) + 0.45, "主智能体可用工具", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#1565c0", transform=ax.transAxes)

    for i, tool in enumerate(tools):
        y = len(tools) - i - 0.15
        ax.text(0.06, y, f"\u2022  {tool}", ha="left", va="center",
                fontsize=11, color="#222", transform=ax.transAxes)

    ax.set_title("主智能体工具集", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "main_agent_tools.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: main_agent_tools.png")


def fig_sim_agent_tools():
    tools = [
        "initialize_simulation  \u2014 环境初始化",
        "create_object  \u2014 创建几何体",
        "push_cube_step  \u2014 步进式推动方块",
        "grab_and_place_step  \u2014 步进式抓取放置",
        "get_object_state  \u2014 查询物体状态",
        "set_object_position  \u2014 设置物体位置",
        "adjust_physics  \u2014 调节物理参数",
        "step_simulation  \u2014 执行仿真步进",
        "cleanup_simulation_tool  \u2014 清理仿真环境",
        "reset_simulation  \u2014 重置仿真世界",
    ]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(tools) + 0.8)
    ax.axis("off")

    bbox = FancyBboxPatch((0.01, 0.05), 0.98, len(tools) + 0.55,
                           boxstyle="round,pad=0.08", facecolor="#ffebee",
                           edgecolor="#e53935", linewidth=2, transform=ax.transAxes)
    ax.add_patch(bbox)
    ax.text(0.5, len(tools) + 0.45, "仿真智能体可用工具（PyBullet）", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#e53935", transform=ax.transAxes)

    for i, tool in enumerate(tools):
        y = len(tools) - i - 0.15
        ax.text(0.04, y, f"\u2022  {tool}", ha="left", va="center",
                fontsize=10, color="#222", transform=ax.transAxes)

    ax.set_title("仿真智能体工具集", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "sim_agent_tools.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: sim_agent_tools.png")


def fig_analysis_agent_tools():
    tools = [
        "质量评分（自有）  \u2014 verdict / quality_score",
        "误差分析（自有）  \u2014 偏差计算 / 来源诊断",
        "策略建议（自有）  \u2014 参数调整 / 策略改进建议",
        "JSON 报告输出（自有）  \u2014 结构化评估报告",
    ]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, len(tools) + 0.8)
    ax.axis("off")

    bbox = FancyBboxPatch((0.01, 0.05), 0.98, len(tools) + 0.55,
                           boxstyle="round,pad=0.08", facecolor="#e8f5e9",
                           edgecolor="#2e7d32", linewidth=2, transform=ax.transAxes)
    ax.add_patch(bbox)
    ax.text(0.5, len(tools) + 0.45, "分析智能体评估能力（自有）", ha="center", va="center",
            fontsize=13, fontweight="bold", color="#2e7d32", transform=ax.transAxes)

    for i, tool in enumerate(tools):
        y = len(tools) - i - 0.15
        ax.text(0.06, y, f"\u2022  {tool}", ha="left", va="center",
                fontsize=11, color="#222", transform=ax.transAxes)

    ax.set_title("分析智能体评估能力", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "analysis_agent_tools.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: analysis_agent_tools.png")


# ═══════════════════════════════════════════════════════════
# 图11：实验三——测试集按难度分布
# ═══════════════════════════════════════════════════════════

def fig_exp3_test_difficulty():
    counts = {"简单 (easy)": 3, "中等 (medium)": 7, "困难 (hard)": 10}
    colors = ["#43a047", "#ffa726", "#ef5350"]

    fig, ax = plt.subplots(figsize=(9, 6))
    bars = ax.bar(list(counts.keys()), list(counts.values()), color=colors,
                  edgecolor="white", linewidth=1.8, width=0.55)
    for bar, val in zip(bars, counts.values()):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                str(val), ha="center", va="bottom", fontsize=15, fontweight="bold")
    ax.set_title("实验三：测试集（20条）按难度分布", fontsize=14, fontweight="bold", pad=12)
    ax.set_ylabel("任务数量（条）", fontsize=12)
    ax.set_ylim(0, max(counts.values()) * 1.25)
    ax.set_xlabel("难度等级", fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_test_difficulty.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: exp3_test_difficulty.png")


# ═══════════════════════════════════════════════════════════
# 图12：实验三——测试集按任务类型分布
# ═══════════════════════════════════════════════════════════

def fig_exp3_test_categories():
    cats = {
        "误差分析": 4,
        "中等搬运": 2,
        "物理参数": 2,
        "安全约束": 2,
        "视觉反馈": 2,
        "单物体搬运": 2,
        "路径规划": 1,
        "多物体操作": 1,
        "协同规划": 1,
        "实时控制": 1,
        "物体创建": 1,
    }

    fig, ax = plt.subplots(figsize=(10, 7))
    names = list(cats.keys())
    vals = list(cats.values())
    colors = plt.cm.Blues(np.linspace(0.35, 0.9, len(names)))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.05, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=11, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.5)
    ax.set_title("实验三：测试集（20条）按任务类型分布", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("任务数量（条）", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_test_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: exp3_test_categories.png")


# ═══════════════════════════════════════════════════════════
# 图13：实验三——训练集（80条）按任务类型分布
# ═══════════════════════════════════════════════════════════

def fig_exp3_train_categories():
    cats = {
        "Gazebo基础操作": 11,
        "错误处理与恢复": 8,
        "物理参数调整": 5,
        "基础搬运": 4,
        "多物体操作": 4,
        "完整实验流程": 4,
        "分步控制": 4,
        "安全约束": 4,
        "路径规划": 3,
        "视觉反馈": 3,
        "环境检查": 3,
        "稳定性测试": 3,
        "异步等待": 3,
        "对比实验": 3,
        "工具选择优化": 3,
        "状态管理": 3,
        "往返误差分析": 3,
        "仿真后分析": 3,
        "口语化混合指令": 2,
        "跨平台对比": 4,
    }

    fig, ax = plt.subplots(figsize=(11, 9))
    names = list(cats.keys())
    vals = list(cats.values())
    colors = plt.cm.Greens(np.linspace(0.3, 0.85, len(names)))
    bars = ax.barh(names, vals, color=colors, edgecolor="white", linewidth=0.8)
    for bar, val in zip(bars, vals):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), va="center", fontsize=9.5, fontweight="bold")
    ax.set_xlim(0, max(vals) * 1.4)
    ax.set_title("实验三：训练集（80条）按任务类型分布", fontsize=14, fontweight="bold", pad=12)
    ax.set_xlabel("任务数量（条）", fontsize=12)
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_train_categories.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: exp3_train_categories.png")


# ═══════════════════════════════════════════════════════════
# 图14：实验三——训练集统计信息表
# ═══════════════════════════════════════════════════════════

def fig_exp3_train_stats():
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.axis("off")

    stats = [
        ("总任务数", "80 条"),
        ("任务类别数", "20 类"),
        ("涵盖仿真引擎", "PyBullet + Gazebo"),
        ("Gazebo 相关任务", "15 条（跨平台对比 4 + Gazebo 11）"),
        ("错误处理任务", "8 条（占总任务 10%）"),
        ("多步骤/分控任务", "11 条（路径+分步+异步）"),
        ("平均每类任务数", "4 条"),
    ]

    cell_data = [[t, d] for t, d in stats]
    tbl = ax.table(
        cellText=cell_data, colLabels=["统计项", "数值"],
        loc="center", cellLoc="left",
        bbox=[0.05, 0.1, 0.9, 0.88],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(12)
    tbl.scale(1.0, 1.9)
    for j in range(2):
        tbl[(0, j)].set_facecolor("#2e7d32")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_data) + 1):
        for j in range(2):
            fc = "#e8f5e9" if i % 2 == 1 else "white"
            tbl[(i, j)].set_facecolor(fc)
            if j == 0:
                tbl[(i, j)].set_text_props(fontweight="bold")

    ax.set_title("训练集统计信息", fontsize=14, fontweight="bold", pad=12)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_train_stats.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: exp3_train_stats.png")


# ═══════════════════════════════════════════════════════════
# 图15：实验三——训练集 Top-5 类别占比饼图
# ═══════════════════════════════════════════════════════════

def fig_exp3_train_pie():
    top5 = [("Gazebo基础操作", 11), ("错误处理与恢复", 8),
            ("物理参数调整", 5), ("基础搬运", 4), ("多物体操作", 4)]
    rest = 80 - sum(v for _, v in top5)
    labels = [n for n, _ in top5] + [f"其他 {20-5} 类"]
    vals = [v for _, v in top5] + [rest]
    colors = list(plt.cm.Greens(np.linspace(0.3, 0.85, 5))) + ["#bdbdbd"]
    explode = [0.06] * 5 + [0]

    fig, ax = plt.subplots(figsize=(9, 8))
    wedges, texts, autotexts = ax.pie(
        vals, labels=None, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for at in autotexts:
        at.set_fontsize(10)
        at.set_fontweight("bold")
        at.set_color("white")
    ax.legend(
        wedges, [f"{l}（{v}条）" for l, v in zip(labels, vals)],
        loc="lower right", fontsize=10.5, framealpha=0.9
    )
    ax.set_title("训练集类别 Top-5 占比（其余 16 类占 44.3%）",
                 fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "exp3_train_pie.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: exp3_train_pie.png")


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("开始生成 charts 图表（全部单独输出）...")
    print("=" * 60)

    # RAG
    fig_rag_doc_counts()
    fig_rag_chunk_counts()
    fig_rag_chunk_pie()

    # MCP
    fig_mcp_tools()

    # 系统总览
    fig_tech_stack()
    fig_agent_roles()
    fig_core_features()

    # 核心机制
    fig_react_loop()
    fig_multi_agent_arch()
    fig_experience_replay()
    fig_data_flow()

    # 工具集
    fig_main_agent_tools()
    fig_sim_agent_tools()
    fig_analysis_agent_tools()

    # 实验三
    fig_exp3_test_difficulty()
    fig_exp3_test_categories()
    fig_exp3_train_categories()
    fig_exp3_train_stats()
    fig_exp3_train_pie()

    print("=" * 60)
    print(f"所有图表已保存至: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
