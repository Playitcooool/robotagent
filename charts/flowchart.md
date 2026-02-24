# 流程图（聊天 + 工具 + 实时仿真）

```mermaid
flowchart TD
  start([开始])
  login{已登录?}
  do_login[登录/注册]
  chat[发送聊天请求]
  auth[后端鉴权]
  call_model[调用模型/代理]
  tool_needed{需要工具?}
  call_tool[调用 MCP 工具]
  write_frames[写入 latest.png/json]
  sse[后端 SSE 推送帧]
  render[前端渲染画面]
  reply[返回聊天回复]
  end([结束])

  start --> login
  login -- 否 --> do_login --> chat
  login -- 是 --> chat
  chat --> auth --> call_model
  call_model --> tool_needed
  tool_needed -- 否 --> reply --> end
  tool_needed -- 是 --> call_tool --> write_frames --> sse --> render --> reply --> end
```
