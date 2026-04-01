# ER 图（Mermaid）

%%{init: {'theme': 'base', 'themeVariables': {'primaryColor': '#E3EDF7', 'primaryTextColor': '#2C3E50', 'primaryBorderColor': '#5D7B9D', 'lineColor': '#5D7B9D', 'fontFamily': 'Arial', 'fontSize': '14px'}}}%%

```mermaid
erDiagram
  USER {
    uuid id PK "主键"
    string username "用户名"
    string password_hash "bcrypt hash"
    timestamp created_at "创建时间 ISO8601"
  }

  SESSION {
    string token PK "Redis KEY: session:{token}"
    uuid user_id FK "关联用户"
    timestamp expires_at "过期时间 ISO8601"
  }

  CHAT_SESSION {
    uuid id PK "主键"
    uuid user_id FK "关联用户"
    timestamp created_at "创建时间 ISO8601"
  }

  CHAT_MESSAGE {
    uuid id PK "主键"
    uuid session_id FK "关联会话"
    string role "user/assistant/system"
    text content "消息内容"
    timestamp created_at "创建时间 ISO8601"
  }

  USER ||--o{ SESSION : "has"
  USER ||--o{ CHAT_SESSION : "owns"
  CHAT_SESSION ||--o{ CHAT_MESSAGE : "contains"
```

说明：后端使用 Redis 存储会话与聊天记录，这里用 ER 形式表达逻辑结构。