# 需求分析图（Mermaid）

%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#E3EDF7', 'primaryTextColor': '#2C3E50', 'primaryBorderColor': '#5D7B9D', 'lineColor': '#5D7B9D', 'fontFamily': 'Arial', 'fontSize': '14px', 'clusterBkg': '#F5F7FA'}}}%%

```mermaid
mindmap
  root((RobotAgent 需求))
    目标
      多代理对话与工具调用
      实时仿真可视化
      Training-Free GRPO 经验积累
    功能性需求
      账号与鉴权
        注册
        登录
        会话保持
      聊天与工具
        聊天消息
        工具调用（仿真/分析）
        结果展示
      仿真流
        MCP 工具执行
        实时帧推送
        右侧面板渲染
      Training-Free GRPO
        轨迹采样与评分
        Best vs Worst 对比
        经验提炼与存储
        Experience 注入子 Agent
      评估实验
        exp1: 学术问答质量评估（BLEU/ROUGE）
        exp3: 尝试次数与成功率分析
        exp4: Experience 效果对比（有无经验库）
    非功能性需求
      性能
        实时帧延迟 < 2s
        聊天响应延迟 < 10s（不含仿真）
        帧率 ≥ 30fps
      可靠性
        Redis 会话存储（持久化）
        MCP 工具超时重试（≤ 3次）
      可配置
        模型与 MCP 配置
        GRPO 超参数调整
      可部署
        前后端分离
        Conda 环境隔离
    约束
      Python 3.10+
      Node.js 18+
      Redis 7.0+
      MCP/后端共享帧目录
      Ollama 本地模型服务
      仿真帧目录: mcp/.sim_stream
```
