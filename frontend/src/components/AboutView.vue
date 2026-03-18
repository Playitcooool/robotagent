<template>
  <div class="about-page">
    <section class="hero">
      <p class="eyebrow">RobotAgent Platform</p>
      <h1>多 Agent + 仿真闭环的机器人任务平台</h1>
      <p class="lead">
        RobotAgent 将大模型任务编排、数据分析、仿真控制和前端可视化合并为一个工作流。
        三个代理（主代理、仿真代理、分析代理）均通过 OMLX 本地部署推理，用户可在同一界面中发起任务、查看执行计划、追踪时间轴，并获得仿真回传画面。
      </p>
      <div class="hero-tags">
        <span>LangChain / ReAct</span>
        <span>OMLX 本地推理</span>
        <span>PyBullet + Gazebo MCP</span>
        <span>Vue 3 + FastAPI</span>
      </div>
    </section>

    <section class="grid two">
      <article class="card">
        <h2>核心能力</h2>
        <ul>
          <li>主代理负责任务理解、路由和结果归纳，不直接做底层仿真。</li>
          <li>分析代理处理数据解读与指标分析任务。</li>
          <li>仿真代理通过 MCP 调用 PyBullet / Gazebo 工具执行场景操作。</li>
          <li>三个代理均使用本地 OMLX 推理服务（MLX-Qwen3.5-4B），无需外部 API。</li>
          <li>前端实时展示 token 输出、planning 更新、timeline 和仿真画面。</li>
        </ul>
      </article>

      <article class="card">
        <h2>系统架构</h2>
        <ol>
          <li>前端发送用户问题到 `/api/chat/send`。</li>
          <li>后端通过 `active_agent.astream(...)` 产生增量事件流。</li>
          <li>事件按 `delta / thinking / planning / timeline / usage` 分类发送。</li>
          <li>仿真结果由 `/api/sim/stream` 和 `/api/sim/latest.png` 持续回传。</li>
          <li>聊天记录与会话索引存入 Redis，支持会话切换和删除。</li>
        </ol>
      </article>
    </section>

    <section class="grid three">
      <article class="card">
        <h3>后端模块</h3>
        <p><code>backend/app.py</code> 负责主入口、流式事件聚合和会话 API。</p>
        <p><code>backend/routes_auth.py</code> 提供登录鉴权能力。</p>
        <p><code>backend/routes_sim.py</code> 暴露仿真帧与调试接口。</p>
        <p><code>backend/stream_utils.py</code> 处理 thinking / planning 解析。</p>
      </article>

      <article class="card">
        <h3>代理与工具</h3>
        <p><code>tools/SubAgentTool.py</code> 初始化分析代理与仿真代理，统一使用本地 OMLX 推理。</p>
        <p><code>tools/AnalysisTool.py</code> 提供数据分析工具集。</p>
        <p><code>mcp/mcp_server.py</code> 是 PyBullet MCP 服务。</p>
        <p><code>mcp/gazebo_mcp_server.py</code> 是 Gazebo MCP 服务。</p>
      </article>

      <article class="card">
        <h3>前端交互</h3>
        <p>首屏为居中输入模式，发送后切换到完整聊天布局。</p>
        <p>Thinking 作为独立折叠组件显示，避免污染最终答案区。</p>
        <p>Planning 单独展示，右栏专注工具结果与仿真画面。</p>
        <p>支持 Markdown / 表格 / 代码块 / 数学公式渲染。</p>
      </article>
    </section>

    <section class="card timeline-card">
      <h2>典型执行流程</h2>
      <div class="flow">
        <div class="flow-step"><span>1</span><p>用户提出任务（例如机械臂抓取或轨迹分析）。</p></div>
        <div class="flow-step"><span>2</span><p>主代理拆解任务并生成简明计划，逐步更新状态。</p></div>
        <div class="flow-step"><span>3</span><p>根据意图调用仿真代理或分析代理执行具体任务。</p></div>
        <div class="flow-step"><span>4</span><p>前端实时接收推理、计划、时间轴和仿真回传帧。</p></div>
        <div class="flow-step"><span>5</span><p>主代理整合结果，输出简洁结论与下一步建议。</p></div>
      </div>
    </section>

    <section class="grid two">
      <article class="card">
        <h2>一键开发启动</h2>
        <pre><code>./dev.sh</code></pre>
        <p>该脚本会同时启动后端（8000）和前端（5173），并清理端口占用，支持反复重启。</p>
      </article>

      <article class="card">
        <h2>仿真服务准备</h2>
        <pre><code>docker compose -f docker/pybullet/docker-compose.yml up -d --build
docker compose -f docker/gazebo/docker-compose.yml up -d --build</code></pre>
        <p>启动后请确认 `config/config.yml` 中 MCP 地址与端口映射一致。</p>
      </article>
    </section>
  </div>
</template>

<script>
export default {
  name: 'AboutView'
}
</script>

<style scoped>
.about-page {
  height: calc(100vh - 58px);
  overflow: auto;
  padding: 18px;
  display: grid;
  gap: 14px;
  scrollbar-width: thin;
  scrollbar-color: rgba(154, 164, 178, 0.45) #151922;
}

.about-page::-webkit-scrollbar { width: 9px; }
.about-page::-webkit-scrollbar-track { background: #151922; border-radius: 999px; }
.about-page::-webkit-scrollbar-thumb {
  background: rgba(154, 164, 178, 0.45);
  border-radius: 999px;
  border: 2px solid #151922;
}

.hero {
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 16px;
  background:
    radial-gradient(1200px 220px at -10% -40%, rgba(47, 125, 255, 0.22), transparent 45%),
    radial-gradient(600px 220px at 110% -40%, rgba(40, 193, 153, 0.14), transparent 50%),
    #141a27;
  padding: 18px;
}

.eyebrow {
  margin: 0 0 8px;
  font-size: 12px;
  letter-spacing: 0.08em;
  text-transform: uppercase;
  color: #9fc1ff;
}

.hero h1 {
  margin: 0;
  font-size: 30px;
  line-height: 1.25;
}

.lead {
  margin: 10px 0 0;
  max-width: 90ch;
  color: #c7d1de;
  line-height: 1.7;
}

.hero-tags {
  display: flex;
  flex-wrap: wrap;
  gap: 8px;
  margin-top: 12px;
}

.hero-tags span {
  border: 1px solid rgba(255, 255, 255, 0.16);
  border-radius: 999px;
  padding: 4px 10px;
  font-size: 12px;
  color: #dbe2ed;
  background: rgba(255, 255, 255, 0.03);
}

.grid {
  display: grid;
  gap: 14px;
}

.grid.two {
  grid-template-columns: repeat(2, minmax(0, 1fr));
}

.grid.three {
  grid-template-columns: repeat(3, minmax(0, 1fr));
}

.card {
  border: 1px solid rgba(255, 255, 255, 0.1);
  border-radius: 14px;
  background: #141925;
  padding: 14px;
}

.card h2,
.card h3 {
  margin: 0 0 10px;
}

.card p {
  margin: 7px 0;
  color: #cad3df;
  line-height: 1.65;
}

.card ul,
.card ol {
  margin: 0;
  padding-left: 1.2em;
  color: #d4dde8;
  line-height: 1.7;
}

.flow {
  display: grid;
  gap: 8px;
}

.flow-step {
  display: grid;
  grid-template-columns: 28px 1fr;
  gap: 10px;
  align-items: start;
  padding: 8px 0;
  border-bottom: 1px dashed rgba(255, 255, 255, 0.08);
}

.flow-step:last-child {
  border-bottom: 0;
}

.flow-step span {
  width: 28px;
  height: 28px;
  border-radius: 999px;
  display: grid;
  place-items: center;
  background: rgba(47, 125, 255, 0.18);
  color: #9fc1ff;
  font-weight: 700;
}

.flow-step p {
  margin: 2px 0 0;
}

pre {
  margin: 0 0 8px;
  border-radius: 10px;
  background: #0d1117;
  border: 1px solid rgba(255, 255, 255, 0.08);
  padding: 10px 12px;
  overflow-x: auto;
}

code {
  font-family: "JetBrains Mono", "Menlo", "Consolas", monospace;
  font-size: 12.5px;
}

@media (max-width: 1080px) {
  .grid.two,
  .grid.three {
    grid-template-columns: 1fr;
  }

  .hero h1 {
    font-size: 24px;
  }
}
</style>
