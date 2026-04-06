#!/usr/bin/env python3
"""
charts/ 生成脚本 —— 绘制论文所需的各类统计图表（中文标签）
直接读取项目数据，无需重新运行实验。
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
# 图1：RAG 知识库来源分布
# ═══════════════════════════════════════════════════════════

def plot_rag_source_distribution():
    """RAG 知识库：文档数 & chunk 数按来源分布"""
    sources = {
        "ROS 2 Humble\n官方文档": {"docs": 6, "chunks": 374, "color": "#1565c0"},
        "PyBullet\n官方论坛": {"docs": 116, "chunks": 47, "color": "#e53935"},
        "ManiSkill\n官方文档": {"docs": 4, "chunks": 31, "color": "#43a047"},
        "Gazebo\n官方文档": {"docs": 4, "chunks": 3, "color": "#fb8c00"},
    }

    names = list(sources.keys())
    doc_counts = [s["docs"] for s in sources.values()]
    chunk_counts = [s["chunks"] for s in sources.values()]
    colors = [s["color"] for s in sources.values()]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle("RAG 知识库来源分布", fontsize=15, fontweight="bold", y=1.01)

    # 左：文档数柱状图
    bars1 = ax1.bar(names, doc_counts, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val in zip(bars1, doc_counts):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax1.set_title("原始文档数（份）", fontsize=12)
    ax1.set_ylabel("文档数量")
    ax1.set_ylim(0, max(doc_counts) * 1.18)

    # 右：Chunk 数柱状图
    bars2 = ax2.bar(names, chunk_counts, color=colors, edgecolor="white", linewidth=1.5, width=0.55)
    for bar, val in zip(bars2, chunk_counts):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                str(val), ha="center", va="bottom", fontsize=12, fontweight="bold")
    ax2.set_title("切分 Chunk 数（块）", fontsize=12)
    ax2.set_ylabel("Chunk 数量")
    ax2.set_ylim(0, max(chunk_counts) * 1.18)

    # 总计标注
    total_chunks = sum(chunk_counts)
    total_docs = sum(doc_counts)
    fig.text(0.5, -0.04, f"合计：{total_docs} 份文档 → {total_chunks} 个 Chunk，平均每份文档 {total_chunks/total_docs:.1f} 块",
             ha="center", fontsize=10, color="#555555", style="italic")

    fig.tight_layout()
    out = OUT_DIR / "rag_source_distribution.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


def plot_rag_chunk_pie():
    """RAG Chunk 数占比饼图"""
    labels = ["ROS 2 Humble", "PyBullet", "ManiSkill", "Gazebo"]
    chunks = [374, 47, 31, 3]
    colors = ["#1565c0", "#e53935", "#43a047", "#fb8c00"]
    explode = [0.05, 0, 0, 0]

    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        chunks, labels=labels, colors=colors, explode=explode,
        autopct="%1.1f%%", startangle=90,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
        textprops={"fontsize": 12},
    )
    for at in autotexts:
        at.set_fontsize(11)
        at.set_fontweight("bold")
        at.set_color("white")
    for text in texts:
        text.set_fontsize(11)

    ax.set_title("RAG Chunk 来源占比", fontsize=14, fontweight="bold", pad=12)
    ax.legend(wedges, [f"{l}（{c}块）" for l, c in zip(labels, chunks)],
               loc="lower right", fontsize=10)

    out = OUT_DIR / "rag_chunk_pie.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图2：MCP 工具集成分类
# ═══════════════════════════════════════════════════════════

def plot_mcp_tools():
    """MCP 工具集分类：PyBullet（14个）+ Gazebo（20个）"""

    pybullet_tools = [
        ("initialize_simulation", "环境初始化"),
        ("check_static_assets", "静态资源检查"),
        ("push_cube_step", "方块推动"),
        ("grab_and_place_step", "抓取放置"),
        ("path_planning", "路径规划"),
        ("adjust_physics", "物理参数调节"),
        ("multi_object_grab_and_place", "多物体协同"),
        ("simulate_vision_sensor", "视觉传感器"),
        ("cleanup_simulation_tool", "环境清理"),
        ("check_simulation_state", "状态查询"),
        ("reset_simulation", "仿真重置"),
        ("pause_simulation", "暂停"),
        ("unpause_simulation", "恢复"),
        ("get_object_state", "物体状态"),
        ("set_object_position", "位置设置"),
        ("step_simulation", "步进仿真"),
        ("create_object", "创建物体"),
        ("delete_object", "删除物体"),
        ("get_simulation_info", "仿真信息"),
        ("set_gravity", "重力设置"),
    ]

    gazebo_tools = [
        ("initialize_ros_connection", "ROS 连接初始化"),
        ("spawn_model", "模型生成"),
        ("list_builtin_models", "内置模型列表"),
        ("delete_model", "删除模型"),
        ("get_model_state", "模型状态查询"),
        ("set_model_state", "模型状态设置"),
        ("list_models", "模型列表"),
        ("pause_simulation", "暂停"),
        ("unpause_simulation", "恢复"),
        ("reset_simulation", "仿真重置"),
        ("reset_world", "世界重置"),
        ("capture_camera", "相机拍摄"),
        ("cleanup_ros_connection", "ROS 连接清理"),
        ("clear_simulation_state", "状态清理"),
        ("get_simulation_info", "仿真信息"),
        ("apply_force", "外力施加"),
        ("move_object", "物体移动"),
        ("create_simple_object", "创建简单物体"),
    ]

    # 分类
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

    def count_category(tool_list, cat_tools):
        return sum(1 for t in tool_list if any(ct in t[0] for ct in cat_tools))

    def get_category_tools(tool_list, cat_tools):
        return [t[1] for t in tool_list if any(ct in t[0] for ct in cat_tools)]

    pb_cats = {cat: count_category(pybullet_tools, cat_tools)
               for cat, cat_tools in categories.items()}
    gaz_cats = {cat: count_category(gazebo_tools, cat_tools)
                 for cat, cat_tools in categories.items()}

    cat_names = list(categories.keys())
    pb_vals = [pb_cats[c] for c in cat_names]
    gaz_vals = [gaz_cats[c] for c in cat_names]

    x = np.arange(len(cat_names))
    width = 0.38

    fig, ax = plt.subplots(figsize=(13, 6))
    bars1 = ax.bar(x - width/2, pb_vals, width, label="PyBullet", color="#e53935", edgecolor="white", linewidth=1.2)
    bars2 = ax.bar(x + width/2, gaz_vals, width, label="Gazebo", color="#1565c0", edgecolor="white", linewidth=1.2)

    for bar, val in zip(bars1, pb_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#c62828")
    for bar, val in zip(bars2, gaz_vals):
        if val > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                    str(val), ha="center", va="bottom", fontsize=10, fontweight="bold", color="#1565c0")

    ax.set_xticks(x)
    ax.set_xticklabels(cat_names, fontsize=11)
    ax.set_ylabel("工具数量（个）")
    ax.set_ylim(0, max(max(pb_vals), max(gaz_vals)) + 3)
    ax.legend(loc="upper right", fontsize=11)

    # 工具数总计
    ax.text(0.02, 0.97, f"PyBullet：{len(pybullet_tools)} 个工具  |  Gazebo：{len(gazebo_tools)} 个工具",
            transform=ax.transAxes, fontsize=10, color="#555", style="italic",
            verticalalignment="top")

    ax.set_title("MCP 工具集按功能分类统计", fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    out = OUT_DIR / "mcp_tools_category.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图3：系统架构总览（多列卡片式）
# ═══════════════════════════════════════════════════════════

def plot_system_overview():
    """系统架构总览：三层架构 + 技术栈"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 6))
    fig.suptitle("系统架构总览", fontsize=15, fontweight="bold", y=1.01)

    # 左：技术栈
    ax = axes[0]
    ax.axis("off")
    tech_stack = [
        ("LLM", "Qwen3.5-9B（本地）"),
        ("框架", "LangChain + ReAct"),
        ("协议", "MCP（Model Context Protocol）"),
        ("向量库", "Qdrant"),
        ("缓存/消息", "Redis"),
        ("前端", "Vue3"),
        ("后端", "FastAPI"),
        ("部署", "Docker Compose"),
        ("模型推理", "LM Studio（本地）"),
    ]
    cell_data = [[t, d] for t, d in tech_stack]
    tbl = ax.table(cellText=cell_data, colLabels=["技术", "选型"],
                   loc="upper center", cellLoc="left", bbox=[0, 0, 1, 1])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.6)
    for j in range(2):
        tbl[(0, j)].set_facecolor("#1565c0")
        tbl[(0, j)].set_text_props(color="white", fontweight="bold")
    for i in range(1, len(cell_data) + 1):
        tbl[(i, 0)].set_facecolor("#e3f2fd")
        tbl[(i, 0)].set_text_props(fontweight="bold")
    ax.set_title("技术栈", fontsize=12, fontweight="bold")

    # 中：智能体角色
    ax = axes[1]
    ax.axis("off")
    agents = [
        ("主智能体", "任务理解、规划协调\nReAct 推理、工具调度", "#1565c0"),
        ("仿真智能体", "执行仿真操作\nPyBullet / Gazebo 调用", "#e53935"),
        ("分析智能体", "结果评估、误差分析\n质量评分、策略建议", "#43a047"),
    ]
    for i, (name, desc, color) in enumerate(agents):
        y = 0.85 - i * 0.32
        bbox = FancyBboxPatch((0.05, y - 0.12), 0.9, 0.26,
                               boxstyle="round,pad=0.02", facecolor=color, alpha=0.15,
                               edgecolor=color, linewidth=2, transform=ax.transAxes)
        ax.add_patch(bbox)
        ax.text(0.5, y, name, ha="center", va="center", fontsize=11,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, y - 0.07, desc, ha="center", va="center", fontsize=8.5,
                color="#333", transform=ax.transAxes, style="italic")
    ax.set_title("三智能体角色", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    # 右：核心特性
    ax = axes[2]
    ax.axis("off")
    features = [
        ("Agentic RAG", "检索按需触发，多路并发召回", "#fb8c00"),
        ("MCP 协议", "工具标准化接入，热插拔工具集", "#8e24aa"),
        ("经验回放", "轨迹收集→筛选→few-shot 注入", "#00838f"),
        ("实时可视化", "SSE 推送 + Canvas 帧渲染", "#ad1457"),
        ("状态外置", "Redis 共享上下文，断电可恢复", "#4e342e"),
    ]
    for i, (name, desc, color) in enumerate(features):
        y = 0.88 - i * 0.2
        bbox = FancyBboxPatch((0.05, y - 0.09), 0.9, 0.17,
                               boxstyle="round,pad=0.02", facecolor=color, alpha=0.12,
                               edgecolor=color, linewidth=1.5, transform=ax.transAxes)
        ax.add_patch(bbox)
        ax.text(0.08, y, name, ha="left", va="center", fontsize=10,
                fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.08, y - 0.055, desc, ha="left", va="center", fontsize=8.5,
                color="#444", transform=ax.transAxes)
    ax.set_title("核心特性", fontsize=12, fontweight="bold")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    fig.tight_layout()
    out = OUT_DIR / "system_overview.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图4：ReAct 推理循环示意图
# ═══════════════════════════════════════════════════════════

def plot_react_loop():
    """ReAct 推理-行动-观察 循环示意图"""
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    nodes = [
        (5, 6.2, "用户指令", "#e3f2fd", "#1565c0"),
        (2.5, 4.5, "Reason\n推理", "#fff3e0", "#e65100"),
        (5, 4.5, "Act\n行动", "#e8f5e9", "#2e7d32"),
        (7.5, 4.5, "Observe\n观察", "#fce4ec", "#ad1457"),
        (5, 2.5, "上下文\n更新", "#f3e5f5", "#6a1b9a"),
        (5, 0.8, "结束 / 输出\n回复", "#ffecb3", "#f57f17"),
    ]

    # 画连接箭头
    arrows = [
        (nodes[0][0], nodes[0][1]-0.25, nodes[1][0], nodes[1][1]+0.3),
        (nodes[1][0]+0.4, nodes[1][1], nodes[2][0]-0.35, nodes[2][1]),
        (nodes[2][0]+0.35, nodes[2][1], nodes[3][0]-0.35, nodes[3][1]),
        (nodes[1][0], nodes[1][1]-0.3, nodes[4][0], nodes[4][1]+0.28),
        (nodes[4][0], nodes[4][1]-0.28, nodes[5][0], nodes[5][1]+0.28),
        (nodes[3][0]-0.3, nodes[3][1]-0.15, nodes[1][0]+0.25, nodes[1][1]+0.2),
    ]
    for x1, y1, x2, y2 in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                   arrowprops=dict(arrowstyle="->", color="#888", lw=1.5))

    # 循环标注
    ax.annotate("", xy=(2.5, 5.2), xytext=(2.5, 3.8),
               arrowprops=dict(arrowstyle="->", color="#e65100", lw=1.8,
                               connectionstyle="arc3,rad=0.5"))
    ax.text(1.5, 4.5, "循环", fontsize=9, color="#e65100", style="italic")

    # 画节点
    for x, y, label, fc, ec in nodes:
        bbox = FancyBboxPatch((x-0.7, y-0.38), 1.4, 0.76,
                                boxstyle="round,pad=0.08", facecolor=fc,
                                edgecolor=ec, linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=ec)

    ax.set_title("ReAct 推理-行动-观察循环", fontsize=14, fontweight="bold", pad=10)

    # 说明文字
    note = ("ReAct 循环：Reason（推理下一步动作）→ Act（调用工具执行）→ Observe（获取执行结果）→ Context 更新 → 循环\n"
            "本系统中：Reason → Act 调用 MCP 工具（PyBullet/Gazebo/检索） → Observe 获得仿真状态 → Context 更新")
    ax.text(5, -0.1, note, ha="center", va="top", fontsize=8.5, color="#555",
            style="italic", transform=ax.transData,
            bbox=dict(boxstyle="round", facecolor="#f5f5f5", edgecolor="#ccc", alpha=0.8))

    fig.tight_layout(rect=[0, 0.1, 1, 1])
    out = OUT_DIR / "react_loop.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图5：多智能体协同架构图
# ═══════════════════════════════════════════════════════════

def plot_multi_agent_architecture():
    """三智能体协同架构：主-仿真-分析，Redis 共享"""
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # 智能体框
    def draw_agent(cx, cy, name, sub, color, w=2.2, h=1.4):
        bbox = FancyBboxPatch((cx-w/2, cy-h/2), w, h,
                                boxstyle="round,pad=0.12", facecolor=color, alpha=0.12,
                                edgecolor=color, linewidth=2.2)
        ax.add_patch(bbox)
        ax.text(cx, cy + 0.2, name, ha="center", va="center", fontsize=12,
                fontweight="bold", color=color)
        ax.text(cx, cy - 0.35, sub, ha="center", va="center", fontsize=8.5,
                color="#333", style="italic")

    draw_agent(2, 5.5, "主智能体", "任务理解 / 规划协调", "#1565c0")
    draw_agent(5.5, 5.5, "仿真智能体", "PyBullet / Gazebo 执行", "#e53935")
    draw_agent(9, 5.5, "分析智能体", "质量评估 / 误差分析", "#2e7d32")

    # Redis 共享存储
    redis_box = FancyBboxPatch((4, 2.8), 3, 1.2,
                                 boxstyle="round,pad=0.1", facecolor="#90a4ae", alpha=0.2,
                                 edgecolor="#546e7a", linewidth=2, linestyle="--")
    ax.add_patch(redis_box)
    ax.text(5.5, 3.4, "Redis", ha="center", va="center", fontsize=12,
            fontweight="bold", color="#37474f")
    ax.text(5.5, 3.0, "上下文共享 / Pub/Sub", ha="center", va="center",
            fontsize=9, color="#555")

    # 用户
    user_box = FancyBboxPatch((0, 5.1), 1.2, 0.8,
                               boxstyle="round,pad=0.08", facecolor="#1565c0", alpha=0.15,
                               edgecolor="#1565c0", linewidth=1.5)
    ax.add_patch(user_box)
    ax.text(0.6, 5.5, "用户", ha="center", va="center", fontsize=10, fontweight="bold", color="#1565c0")

    # 箭头
    # 用户 → 主智能体
    ax.annotate("", xy=(0.95, 5.5), xytext=(1.2, 5.5),
               arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.8))
    ax.text(1.05, 5.75, "指令", fontsize=8, color="#1565c0")

    # 主智能体 ↔ 仿真智能体（双向）
    ax.annotate("", xy=(3.35, 5.5), xytext=(4.35, 5.5),
               arrowprops=dict(arrowstyle="<->", color="#546e7a", lw=1.8))
    ax.text(3.85, 5.75, "MCP", fontsize=8, color="#546e7a")

    # 主智能体 ↔ 分析智能体
    ax.annotate("", xy=(6.9, 5.5), xytext=(7.9, 5.5),
               arrowprops=dict(arrowstyle="<->", color="#546e7a", lw=1.8))
    ax.text(7.4, 5.75, "评估", fontsize=8, color="#546e7a")

    # 主智能体 → Redis
    ax.annotate("", xy=(5.5, 4.0), xytext=(3.0, 4.9),
               arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.5,
                                connectionstyle="arc3,rad=0.2"))
    # 仿真智能体 ↔ Redis
    ax.annotate("", xy=(5.5, 3.35), xytext=(5.5, 4.85),
               arrowprops=dict(arrowstyle="<->", color="#78909c", lw=1.5))
    # 分析智能体 → Redis
    ax.annotate("", xy=(5.5, 4.0), xytext=(8.2, 4.9),
               arrowprops=dict(arrowstyle="->", color="#78909c", lw=1.5,
                                connectionstyle="arc3,rad=-0.2"))

    # MCP / 仿真引擎
    ax.text(5.5, 1.6, "MCP 协议", ha="center", va="center", fontsize=11, fontweight="bold", color="#546e7a")
    ax.text(5.5, 1.2, "PyBullet 仿真引擎  |  Gazebo 仿真引擎", ha="center", va="center", fontsize=9, color="#777")
    ax.annotate("", xy=(5.5, 2.7), xytext=(5.5, 1.75),
               arrowprops=dict(arrowstyle="->", color="#90a4ae", lw=1.3))

    ax.set_title("三智能体协同架构", fontsize=14, fontweight="bold", pad=10)

    fig.tight_layout()
    out = OUT_DIR / "multi_agent_arch.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图6：经验回放三环节流程
# ═══════════════════════════════════════════════════════════

def plot_experience_replay():
    """经验回放：收集→筛选→注入 三阶段"""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 5)
    ax.axis("off")

    stages = [
        (1.5, 3.0, "1. 轨迹收集", ["自动记录每次任务执行", "prompt / response", "tool_calls 历史"],
         "#1565c0", "#e3f2fd"),
        (5.5, 3.0, "2. 经验筛选", ["按智能体类型过滤", "分析智能体质量评分", "Jaccard 相似度去重"],
         "#e65100", "#fff3e0"),
        (9.5, 3.0, "3. Few-Shot 注入", ["TopK 筛选高质量经验", "格式化 few-shot 示例", "注入新任务提示词"],
         "#2e7d32", "#e8f5e9"),
    ]

    for cx, cy, title, lines, color, fc in stages:
        bbox = FancyBboxPatch((cx - 1.3, cy - 1.1), 2.6, 2.2,
                                boxstyle="round,pad=0.15", facecolor=fc,
                                edgecolor=color, linewidth=2.2)
        ax.add_patch(bbox)
        ax.text(cx, cy + 0.8, title, ha="center", va="center", fontsize=11,
                fontweight="bold", color=color)
        for i, line in enumerate(lines):
            ax.text(cx, cy + 0.2 - i * 0.55, line, ha="center", va="center",
                    fontsize=9, color="#333")

    # 箭头
    ax.annotate("", xy=(4.25, 3.0), xytext=(2.85, 3.0),
               arrowprops=dict(arrowstyle="->", color="#888", lw=2))
    ax.annotate("", xy=(8.25, 3.0), xytext=(6.85, 3.0),
               arrowprops=dict(arrowstyle="->", color="#888", lw=2))

    # 公式
    ax.text(5.5, 1.1, "Score(e|q) = α · cos_sim(v_q, v_e) + β · Quality(e)        注入经验: E_inject = TopK(Score, k=3)",
             ha="center", va="center", fontsize=9, color="#444",
             bbox=dict(boxstyle="round,pad=0.3", facecolor="#fafafa", edgecolor="#ccc"))

    ax.set_title("经验回放机制：收集 → 筛选 → 注入", fontsize=14, fontweight="bold", pad=8)

    fig.tight_layout()
    out = OUT_DIR / "experience_replay_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图7：系统数据流总览
# ═══════════════════════════════════════════════════════════

def plot_data_flow():
    """系统端到端数据流：用户→前端→后端→多智能体→MCP→仿真"""
    fig, ax = plt.subplots(figsize=(13, 6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 6)
    ax.axis("off")

    nodes = [
        (1.2, 3.0, "用户\n自然语言指令", "#1565c0", "#e3f2fd"),
        (3.5, 3.0, "Vue3 前端\n任务创建 + 可视化", "#6a1b9a", "#f3e5f5"),
        (5.8, 3.0, "FastAPI 后端\nREST API + SSE", "#00838f", "#e0f7fa"),
        (8.1, 3.0, "多智能体层\n主/仿真/分析智能体", "#e65100", "#fff3e0"),
        (10.4, 3.0, "MCP 协议\n工具网关", "#2e7d32", "#e8f5e9"),
        (12.2, 3.0, "仿真引擎\nPyBullet/Gazebo", "#c62828", "#ffebee"),
    ]

    for x, y, label, ec, fc in nodes:
        bbox = FancyBboxPatch((x - 1.0, y - 0.65), 2.0, 1.3,
                                boxstyle="round,pad=0.1", facecolor=fc,
                                edgecolor=ec, linewidth=2)
        ax.add_patch(bbox)
        ax.text(x, y, label, ha="center", va="center", fontsize=9,
                fontweight="bold", color=ec)

    # 箭头
    for i in range(len(nodes) - 1):
        x1 = nodes[i][0] + 1.0
        x2 = nodes[i+1][0] - 1.0
        y = nodes[i][1]
        ax.annotate("", xy=(x2, y), xytext=(x1, y),
                   arrowprops=dict(arrowstyle="->", color="#888", lw=1.8))

    # 中间件标注
    ax.text(4.65, 3.75, "Redis\n上下文共享", ha="center", va="center", fontsize=8,
            color="#546e7a", style="italic")
    ax.text(7.0, 3.75, "Qdrant\n向量检索", ha="center", va="center", fontsize=8,
            color="#546e7a", style="italic")

    # 反馈回路
    ax.annotate("", xy=(5.8, 1.2), xytext=(10.4, 1.2),
               arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.5,
                               connectionstyle="arc3,rad=0"))
    ax.annotate("", xy=(3.5, 2.35), xytext=(5.8, 2.35),
               arrowprops=dict(arrowstyle="<-", color="#c62828", lw=1.5,
                               connectionstyle="arc3,rad=0"))
    ax.text(7.1, 1.35, "SSE 仿真帧流 / 分析评估结果", ha="center", va="center",
            fontsize=8.5, color="#c62828", style="italic")

    ax.set_title("系统端到端数据流", fontsize=14, fontweight="bold", pad=8)

    fig.tight_layout()
    out = OUT_DIR / "data_flow.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 图8：智能体工具调用流程（按智能体分类的工具调用时序）
# ═══════════════════════════════════════════════════════════

def plot_agent_tool_calls():
    """三个智能体各自调用的工具集"""
    fig, axes = plt.subplots(1, 3, figsize=(14, 7))
    fig.suptitle("各智能体可用工具集", fontsize=15, fontweight="bold", y=1.01)

    main_tools = [
        "ReAct 推理（自有）", "任务分解（自有）",
        "search（检索工具）", "MCP 工具调度（间接）",
    ]

    sim_tools = [
        "initialize_simulation", "create_object", "push_cube_step",
        "grab_and_place_step", "get_object_state", "set_object_position",
        "step_simulation", "adjust_physics", "cleanup_simulation_tool",
        "reset_simulation", "get_simulation_info", "set_gravity",
    ]

    analysis_tools = [
        "质量评分（自有）", "误差分析（自有）",
        "策略建议生成（自有）", "JSON 报告输出（自有）",
    ]

    agent_configs = [
        (axes[0], main_tools, "#1565c0", "#e3f2fd", "主智能体\n(MCP Client)"),
        (axes[1], sim_tools, "#e53935", "#ffebee", "仿真智能体\n(PyBullet/Gazebo)"),
        (axes[2], analysis_tools, "#2e7d32", "#e8f5e9", "分析智能体\n(自有评估逻辑)"),
    ]

    for ax, tools, color, fc, title in agent_configs:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, len(tools) + 1)
        ax.axis("off")

        bbox = FancyBboxPatch((0.01, 0.05), 0.98, 0.85,
                               boxstyle="round,pad=0.05", facecolor=fc,
                               edgecolor=color, linewidth=2, transform=ax.transAxes)
        ax.add_patch(bbox)

        ax.text(0.5, len(tools) + 0.35, title, ha="center", va="center",
                fontsize=11, fontweight="bold", color=color, transform=ax.transAxes)

        for i, tool in enumerate(tools):
            y = len(tools) - i - 0.2
            ax.text(0.06, y, f"• {tool}", ha="left", va="center",
                    fontsize=9.5, color="#222", transform=ax.transAxes)

    fig.tight_layout()
    out = OUT_DIR / "agent_tools.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ═══════════════════════════════════════════════════════════
# 主函数
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 60)
    print("开始生成 charts 图表...")
    print("=" * 60)

    plot_rag_source_distribution()
    plot_rag_chunk_pie()
    plot_mcp_tools()
    plot_system_overview()
    plot_react_loop()
    plot_multi_agent_architecture()
    plot_experience_replay()
    plot_data_flow()
    plot_agent_tool_calls()

    print("=" * 60)
    print(f"所有图表已保存至: {OUT_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
