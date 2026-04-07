# 第四章内容补全实现计划

> **For agentic workers:** 使用 superpowers:subagent-driven-development (推荐) 或 superpowers:executing-plans 来逐任务执行。

**目标：** 修改 `2236127阮炜慈初稿_更新版.docx`，补全第四章两处内容：（1）重构4.5节，增加前端渲染与后端架构细节；（2）补全4.6节GRPO pipeline完整描述。

**方法：** 使用 Python（lxml + zipfile）直接编辑 OpenXML 结构，替换和插入段落、表格内容。

**技术栈：** Python 3, lxml, python-docx（仅用于读取结构，实际写入用底层OpenXML），原文档已有样式模板保持不变。

---

## 文件修改概览

| 操作 | 段落位置 | 内容 |
|------|---------|------|
| 替换 | 474（界面模块段落） | 扩充界面模块描述 |
| 替换 | 476（状态管理段落） | 扩充状态管理描述 |
| 替换 | 479-480（原状态管理内容） | 扩充状态管理描述 |
| 插入 | 4.5.2后新增 | 4.5.3 前端仿真渲染技术 |
| 插入 | 4.5.3后新增 | 4.5.4 后端架构与异步请求 |
| 替换 | 482（4.6现有句段） | 补全4.6完整pipeline |

---

## Task 1: 探索文档结构，确认段落索引

**目标：** 精确定位需要修改的段落索引，并读取现有段落内容。

**文件：**
- 读取: `2236127阮炜慈初稿_更新版.docx` → `word/document.xml`

**步骤：**

- [ ] **Step 1: 读取文档，找到所有章节标题段落**

```python
import zipfile
from lxml import etree

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'
DOCX = '/Volumes/Samsung/Projects/robotagent/2236127阮炜慈初稿_更新版.docx'

with zipfile.ZipFile(DOCX, 'r') as z:
    xml_data = z.read('word/document.xml')

root = etree.fromstring(xml_data)
body = root.find(f'.//{{{W_NS}}}body')

# 按body直系子元素索引（表格会占位）
body_children = list(body)
para_indices = {}

for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if '4.5.1' in text and len(text) < 30:
            para_indices['4.5.1_heading'] = i
        if '4.5.2' in text and len(text) < 30:
            para_indices['4.5.2_heading'] = i
        if '4.5.3' in text and len(text) < 30:
            para_indices['4.5.3_heading'] = i
        if '4.6' in text and 'Training-Free' in text and len(text) < 40:
            para_indices['4.6_heading'] = i
        if '4.6' in text and 'Training-Free' not in text and len(text) < 20:
            para_indices['4.6_heading'] = i
        if '本实验设计了Training-free GRPO的pipeline' in text:
            para_indices['4.6_pipeline_start'] = i

print("Found indices:", para_indices)

# 打印4.5节附近body索引内容（前后各5个）
for key in ['4.5.1_heading', '4.5.2_heading', '4.6_heading', '4.6_pipeline_start']:
    if key in para_indices:
        idx = para_indices[key]
        print(f"\n=== {key} at index {idx} ===")
        for j in range(max(0, idx-2), min(len(body_children), idx+8)):
            child = body_children[j]
            tag = child.tag.replace(f'{{{W_NS}}}', '')
            if tag == 'p':
                texts2 = child.findall(f'.//{{{W_NS}}}t')
                t2 = ''.join([t.text or '' for t in texts2])[:80]
                print(f"  [{j}] <p> {t2}")
            elif tag == 'tbl':
                print(f"  [{j}] <tbl>")
```

预期输出：找到 `4.5.1_heading`、`4.5.2_heading`、`4.6_heading`、`4.6_pipeline_start` 的body直系子元素索引。

---

## Task 2: 编写内容填充函数

**目标：** 编写Python脚本生成新的段落XML元素，包含正确的样式属性（宋体、12pt、首行缩进）。

**文件：**
- 创建: `/Volumes/Samsung/Projects/robotagent/chapter4_filler.py`

**步骤：**

- [ ] **Step 1: 创建段落填充函数**

```python
import re
from lxml import etree

W_NS = 'http://schemas.openxmlformats.org/wordprocessingml/2006/main'

def w(tag):
    return f'{{{W_NS}}}{tag}'

def create_body_para(text, bold=False, center=False, first_line_indent=True, font_size='24', font='宋体'):
    """创建正文段落（带首行缩进）"""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    if center:
        jc = etree.SubElement(pPr, w('jc'))
        jc.set(w('val'), 'center')
    else:
        ind = etree.SubElement(pPr, w('ind'))
        ind.set(w('firstLineChars'), '200')
        ind.set(w('firstLine'), '480')
    r = etree.SubElement(p, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), font)
    rFonts.set(w('ascii'), font)
    if bold:
        etree.SubElement(rPr, w('b'))
        etree.SubElement(rPr, w('bCs'))
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), font_size)
    szCs = etree.SubElement(rPr, w('szCs'))
    szCs.set(w('val'), font_size)
    t = etree.SubElement(r, w('t'))
    t.text = text
    if text.startswith(' ') or text.endswith(' '):
        t.set('{http://www.w3.org/XML/1998/namespace}space', 'preserve')
    return p

def create_heading_para(text, level=3):
    """创建章节标题段落（无首行缩进，加粗）"""
    p = etree.Element(w('p'))
    pPr = etree.SubElement(p, w('pPr'))
    outlineLvl = etree.SubElement(pPr, w('outlineLvl'))
    outlineLvl.set(w('val'), str(level))  # 3级标题
    keepNext = etree.SubElement(pPr, w('keepNext'))
    keepNext.set(w('val'), '1')
    r = etree.SubElement(p, w('r'))
    rPr = etree.SubElement(r, w('rPr'))
    rFonts = etree.SubElement(rPr, w('rFonts'))
    rFonts.set(w('eastAsia'), '黑体')
    rFonts.set(w('ascii'), '黑体')
    etree.SubElement(rPr, w('b'))
    sz = etree.SubElement(rPr, w('sz'))
    sz.set(w('val'), '28')  # 14pt
    szCs = etree.SubElement(rPr, w('szCs'))
    szCs.set(w('val'), '28')
    t = etree.SubElement(r, w('t'))
    t.text = text
    return p
```

- [ ] **Step 2: 运行测试验证函数**

```bash
python3 -c "
from chapter4_filler import create_body_para, create_heading_para
from lxml import etree

p = create_heading_para('4.5.3 前端仿真渲染技术', level=3)
print(etree.tostring(p, pretty_print=True).decode()[:500])
"
```

预期输出：生成正确的标题XML，带 `outlineLvl`、`keepNext`、黑体、14pt属性。

- [ ] **Step 3: 提交**

```bash
git add chapter4_filler.py
git commit -m "chore: add chapter4 content filler functions"
```

---

## Task 3: 替换4.5.1（界面模块）内容

**目标：** 找到body中 `4.5.1` 标题段落后紧跟的描述段落，将其替换为扩充后的界面模块描述。

**文件：**
- 修改: `2236127阮炜慈初稿_更新版.docx`

**新内容：**

> 4.5.1 界面模块
>
> 前端界面由认证视图、会话侧栏、对话主视图与结果面板四部分组成，各模块承担独立职责，通过统一状态接口实现协同。认证视图（AuthView）提供用户登录与注册功能，表单校验与错误提示由前端独立处理，降低与后端的无效交互次数。会话侧栏（SessionSidebar）管理会话列表，支持按时间排序的会话切换、创建与删除操作，用户可在侧栏直接发起新任务或回溯历史会话。
>
> 对话主视图（ChatMainView）承担最核心的交互职责，包含消息列表渲染、流式文本接收与展示、工具调用状态指示器三部分。消息列表基于虚拟滚动（Virtual Scrolling）实现，即使在长对话场景下也能保持渲染性能；流式文本通过 `fetch` + `ReadableStream` 消费后端NDJSON响应，逐token追加显示，无需等待完整响应；工具调用状态指示器实时反映当前Agent的工具调用进展，并在工具输出时在消息气泡内渲染结构化JSON。
>
> 结果面板（ResultPanel）包含仿真帧画布、轨迹回放控制与数据导出按钮。仿真帧画布采用 Canvas 2D 方式渲染 PNG 图像，支持自适应缩放与原始分辨率双模式显示；轨迹回放控制条允许用户拖拽到任意中间步骤，观察环境状态随时间的变化；数据导出按钮支持将当前会话的完整轨迹以 JSON 格式下载本地。

**步骤：**

- [ ] **Step 1: 找到4.5.1标题在body_children中的索引**

```python
body_children = list(body)
idx_451 = None
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if text.strip() == '4.5.1 界面模块':
            idx_451 = i
            break
print(f"4.5.1 heading at body index: {idx_451}")
```

- [ ] **Step 2: 找到4.5.1和4.5.2标题之间的段落（界面描述内容）**

```python
idx_452 = None
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if text.strip() == '4.5.2 状态管理':
            idx_452 = i
            break

print(f"4.5.2 heading at: {idx_452}")
# 界面描述段落在 idx_451+1 到 idx_452-1 之间
desc_paragraphs = []
for i in range(idx_451 + 1, idx_452):
    child = body_children[i]
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if text.strip():
            desc_paragraphs.append((i, child, text))
            print(f"  Para at {i}: {text[:60]}")
```

- [ ] **Step 3: 删除旧描述段落，插入新内容**

```python
# 删除旧的描述段落（保留4.5.1标题）
for i, elem, _ in reversed(desc_paragraphs):
    body.remove(elem)

# 重新获取body_children（删除后索引已变化）
body_children = list(body)
# 找到当前4.5.1标题位置
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if text.strip() == '4.5.1 界面模块':
            idx_451_new = i
            break

# 在4.5.1标题后插入新段落
sections_451 = [
    create_body_para('前端界面由认证视图、会话侧栏、对话主视图与结果面板四部分组成，各模块承担独立职责，通过统一状态接口实现协同。认证视图（AuthView）提供用户登录与注册功能，表单校验与错误提示由前端独立处理，降低与后端的无效交互次数。'),
    create_body_para('会话侧栏（SessionSidebar）管理会话列表，支持按时间排序的会话切换、创建与删除操作，用户可在侧栏直接发起新任务或回溯历史会话。'),
    create_body_para('对话主视图（ChatMainView）承担最核心的交互职责，包含消息列表渲染、流式文本接收与展示、工具调用状态指示器三部分。消息列表基于虚拟滚动（Virtual Scrolling）实现，即使在长对话场景下也能保持渲染性能；流式文本通过fetch + ReadableStream消费后端NDJSON响应，逐token追加显示，无需等待完整响应；工具调用状态指示器实时反映当前Agent的工具调用进展，并在工具输出时在消息气泡内渲染结构化JSON。'),
    create_body_para('结果面板（ResultPanel）包含仿真帧画布、轨迹回放控制与数据导出按钮。仿真帧画布采用Canvas 2D方式渲染PNG图像，支持自适应缩放与原始分辨率双模式显示；轨迹回放控制条允许用户拖拽到任意中间步骤，观察环境状态随时间的变化；数据导出按钮支持将当前会话的完整轨迹以JSON格式下载本地。'),
]

for j, para in enumerate(sections_451):
    body.insert(idx_451_new + 1 + j, para)
```

- [ ] **Step 4: 保存并验证**

```python
doc_xml_final = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
file_contents['word/document.xml'] = doc_xml_final

with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, data in file_contents.items():
        zout.writestr(name, data)
print("Saved!")
```

验证：
```python
# 重新读取，验证4.5.1内容
body_children = list(body)
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if 'AuthView' in text or 'ChatMainView' in text or 'ResultPanel' in text:
            print(f"Found new content at {i}: {text[:60]}")
```

- [ ] **Step 5: 提交**

```bash
git add chapter4_filler.py  # update
git commit -m "feat(doc): expand 4.5.1 UI module section with component details"
```

---

## Task 4: 替换4.5.2（状态管理）内容

**目标：** 替换4.5.2标题后的状态管理描述，扩充并发流管理和前端帧缓存内容。

**新内容：**

> 4.5.2 状态管理
>
> 前端根状态统一管理会话状态、对话流状态与仿真流状态，并在消息发送后并发启动文本流与仿真流，确保"语言结果"与"环境反馈"同步到达。状态管理采用"中心状态 + 组件渲染"模式：核心状态集中维护，组件按职责订阅和展示，避免多组件重复管理同一状态带来的竞争问题。
>
> 多事件流并发管理是本系统的核心挑战之一。系统同时存在两条数据流：来自后端LangGraph Agent的LLM文本流（通过fetch + ReadableStream消费NDJSON格式增量文本）和来自后端MCP仿真服务的SSE帧流（通过EventSource接收帧元数据通知）。前端在用户发送消息后同时启动两个异步消费循环，两者独立运行，通过共享的会话ID关联状态。当其中一条流发生错误时，另一条流不受影响，系统可独立进行错误恢复或降级。
>
> 前端帧缓存与去重机制避免重复渲染。仿真帧流推送频率（50ms/200ms/500ms）高于前端渲染帧率，相同 `frame_id` 的帧会被前端缓存并跳过重复绘制。帧缓存使用 `Map<frame_id, ImageBitmap>` 结构，内存上限为50帧，超出后按LRU策略淘汰。当检测到后端SSE连接断开时，前端自动切换为指数退避重连（基础间隔1s，最大间隔30s），并在界面上显示"重连中"提示。

**步骤：**

- [ ] **Step 1: 找到4.5.2和4.6之间的段落**

```python
body_children = list(body)
idx_452 = None
idx_456 = None
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if text.strip() == '4.5.2 状态管理':
            idx_452 = i
        if '4.6 Training-Free GRPO' in text.strip() and len(text.strip()) < 40:
            idx_456 = i

print(f"4.5.2 at {idx_452}, 4.6 at {idx_456}")
# 中间段落
for i in range(idx_452 + 1, idx_456):
    child = body_children[i]
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        print(f"  [{i}] {text[:60]}")
```

- [ ] **Step 2: 删除旧段落，插入新内容（方法同Task 3）**

新段落列表：
```python
sections_452 = [
    create_body_para('前端根状态统一管理会话状态、对话流状态与仿真流状态，并在消息发送后并发启动文本流与仿真流，确保"语言结果"与"环境反馈"同步到达。状态管理采用"中心状态 + 组件渲染"模式：核心状态集中维护，组件按职责订阅和展示，避免多组件重复管理同一状态带来的竞争问题。'),
    create_body_para('多事件流并发管理是本系统的核心挑战之一。系统同时存在两条数据流：来自后端LangGraph Agent的LLM文本流（通过fetch + ReadableStream消费NDJSON格式增量文本）和来自后端MCP仿真服务的SSE帧流（通过EventSource接收帧元数据通知）。前端在用户发送消息后同时启动两个异步消费循环，两者独立运行，通过共享的会话ID关联状态。当其中一条流发生错误时，另一条流不受影响，系统可独立进行错误恢复或降级。'),
    create_body_para('前端帧缓存与去重机制避免重复渲染。仿真帧流推送频率（50ms/200ms/500ms）高于前端渲染帧率，相同frame_id的帧会被前端缓存并跳过重复绘制。帧缓存使用Map<frame_id, ImageBitmap>结构，内存上限为50帧，超出后按LRU策略淘汰。当检测到后端SSE连接断开时，前端自动切换为指数退避重连（基础间隔1s，最大间隔30s），并在界面上显示"重连中"提示。'),
]
```

- [ ] **Step 3: 提交**

```bash
git commit -m "feat(doc): expand 4.5.2 state management with concurrency and caching details"
```

---

## Task 5: 新增4.5.3（前端仿真渲染技术）

**目标：** 在4.5.2标题段落后紧跟的位置插入4.5.3小节（先确认4.5.2现有位置）。

**前置条件：** Task 4完成后，4.5.2在body中的索引已更新。

**新内容：**

> 4.5.3 前端仿真渲染技术
>
> 仿真结果的实时可视化是本系统的关键特性之一。前端通过"元数据通知 + 帧图片轮询"的组合模式实现仿真画面渲染，避免通过SSE直接传输二进制图像数据带来的性能开销。
>
> Canvas 2D直接渲染是本系统的图像渲染方案。前端周期性轮询后端帧接口（`GET /api/sim/frames`），获取最新仿真帧PNG图片，通过创建 `Image` 对象并调用 `canvas.drawImage()` 将图像绘制到 `<canvas>` 元素上。响应式画布根据容器宽度自动缩放，同时在底部状态栏显示当前帧的原始分辨率（320×240）。渲染循环使用 `requestAnimationFrame` 配合 `lastRenderTime` 节流阀（throttle interval = 100ms），在保证流畅视觉更新（60fps峰值）的同时，避免CPU过度占用。
>
> 帧更新协调机制保证前端展示与环境实际状态同步。后端SSE流（`/api/sim/stream`）推送帧元数据（`{timestamp, step_index, task, done}`），前端监听SSE事件并解析元数据，当 `timestamp > lastRenderedTs` 时才触发新的帧轮询请求。帧元数据还用于在SSE断开期间通过HTTP长轮询兜底获取最新帧，保证用户在网络波动时仍能看到一定程度的画面更新。
>
> 断线重连采用指数退避加随机抖动（exponential backoff with jitter）策略。初始重连间隔为1s，每次重连失败后间隔翻倍，最大不超过30s；每次重连前在间隔基础上增加±500ms随机抖动，避免多客户端同时发起重连造成后端涌压。Vue组件通过 `onUnmounted` 生命周期清理SSE连接和轮询定时器，防止组件卸载后的后台异步操作导致内存泄漏。

**步骤：**

- [ ] **Step 1: 在Task 4完成后，找到4.5.2描述内容后的位置插入4.5.3标题和新段落**

```python
body_children = list(body)
# 找到4.5.2的状态管理描述内容之后的位置（即下一标题前）
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if '4.6 Training-Free' in text.strip() and len(text.strip()) < 40:
            insert_idx = i
            break

print(f"Insert 4.5.3 before index: {insert_idx}")

sections_453 = [
    create_heading_para('4.5.3 前端仿真渲染技术', level=3),
    create_body_para('仿真结果的实时可视化是本系统的关键特性之一。前端通过"元数据通知 + 帧图片轮询"的组合模式实现仿真画面渲染，避免通过SSE直接传输二进制图像数据带来的性能开销。'),
    create_body_para('Canvas 2D直接渲染是本系统的图像渲染方案。前端周期性轮询后端帧接口（GET /api/sim/frames），获取最新仿真帧PNG图片，通过创建Image对象并调用canvas.drawImage()将图像绘制到<canvas>元素上。响应式画布根据容器宽度自动缩放，同时在底部状态栏显示当前帧的原始分辨率（320×240）。渲染循环使用requestAnimationFrame配合lastRenderTime节流阀（throttle interval = 100ms），在保证流畅视觉更新（60fps峰值）的同时，避免CPU过度占用。'),
    create_body_para('帧更新协调机制保证前端展示与环境实际状态同步。后端SSE流（/api/sim/stream）推送帧元数据（{timestamp, step_index, task, done}），前端监听SSE事件并解析元数据，当timestamp > lastRenderedTs时才触发新的帧轮询请求。帧元数据还用于在SSE断开期间通过HTTP长轮询兜底获取最新帧，保证用户在网络波动时仍能看到一定程度的画面更新。'),
    create_body_para('断线重连采用指数退避加随机抖动（exponential backoff with jitter）策略。初始重连间隔为1s，每次重连失败后间隔翻倍，最大不超过30s；每次重连前在间隔基础上增加±500ms随机抖动，避免多客户端同时发起重连造成后端涌压。Vue组件通过onUnmounted生命周期清理SSE连接和轮询定时器，防止组件卸载后的后台异步操作导致内存泄漏。'),
]

for j, elem in enumerate(sections_453):
    body.insert(insert_idx + j, elem)
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(doc): add 4.5.3 frontend simulation rendering techniques"
```

---

## Task 6: 新增4.5.4（后端架构与异步请求）

**目标：** 在4.5.3后插入4.5.4后端架构章节。

**新内容：**

> 4.5.4 后端架构与异步请求
>
> 系统后端采用分层架构设计，自上而下分为接入层、服务层、智能层与资源层四层。接入层由Vue3前端组成，通过HTTP/WebSocket与后端通信；服务层基于FastAPI构建，负责鉴权、会话管理、聊天接口与仿真流接口；智能层包含LangGraph主代理与子代理（仿真代理、数据分析代理），通过ReAct循环驱动任务执行；资源层包括Redis（会话存储）、Qdrant（向量知识库）、MCP仿真服务（PyBullet/Gazebo）。
>
> 全链路异步请求处理是保障系统高并发的核心。后端所有API端点均采用FastAPI的`async def`异步函数定义，数据库与缓存操作使用异步客户端（`redis.asyncio`）。在LLM推理链路中，`active_agent.astream()`方法以异步生成器方式返回token事件，无需为每个请求分配独立线程，从而在单进程内支持数百并发连接。请求超时控制通过`asyncio.wait_for()`与`asyncio.timeout()`实现，当客户端断开连接（`request.is_disconnected()`）时立即取消正在进行的操作，释放计算资源。
>
> 流式响应机制分为LLM文本流与仿真帧流两条独立通道。LLM文本流使用HTTP分块传输（Chunked Transfer Encoding），后端以NDJSON格式逐token返回 `{type, content, tool_call}` 结构，前端通过 `fetch().body.getReader()` 消费 `ReadableStream`。仿真帧流采用SSE（Server-Sent Events），后端端点 `/api/sim/stream` 以固定频率（50ms/200ms/500ms，自适应）推送帧元数据，前端EventSource监听 `frame` 事件并触发帧图片的HTTP轮询。这两条流在用户发送消息时并发启动，通过共享的会话ID实现语义关联。
>
> 仿真帧的写入与读取采用原子操作保证数据一致性。MCP仿真服务（PyBullet/Gazebo）将渲染帧写入共享目录（`mcp/.sim_stream/latest.png`），使用"写临时文件 + `os.replace()`"的原子替换模式，避免后端在写入过程中读到半帧数据。FastAPI直接通过文件系统服务PNG文件（`/api/sim/frames`），前端轮询间隔根据仿真状态动态调整：仿真运行中50ms轮询、空闲200ms、空闲超200步则降为500ms，降低空闲期的后端负载。

**步骤：**

- [ ] **Step 1: 在4.5.3内容后、4.6前插入4.5.4**

```python
body_children = list(body)
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if '4.6 Training-Free' in text.strip() and len(text.strip()) < 40:
            insert_idx = i
            break

sections_454 = [
    create_heading_para('4.5.4 后端架构与异步请求', level=3),
    create_body_para('系统后端采用分层架构设计，自上而下分为接入层、服务层、智能层与资源层四层。接入层由Vue3前端组成，通过HTTP/WebSocket与后端通信；服务层基于FastAPI构建，负责鉴权、会话管理、聊天接口与仿真流接口；智能层包含LangGraph主代理与子代理（仿真代理、数据分析代理），通过ReAct循环驱动任务执行；资源层包括Redis（会话存储）、Qdrant（向量知识库）、MCP仿真服务（PyBullet/Gazebo）。'),
    create_body_para('全链路异步请求处理是保障系统高并发的核心。后端所有API端点均采用FastAPI的async def异步函数定义，数据库与缓存操作使用异步客户端（redis.asyncio）。在LLM推理链路中，active_agent.astream()方法以异步生成器方式返回token事件，无需为每个请求分配独立线程，从而在单进程内支持数百并发连接。请求超时控制通过asyncio.wait_for()与asyncio.timeout()实现，当客户端断开连接（request.is_disconnected()）时立即取消正在进行的操作，释放计算资源。'),
    create_body_para('流式响应机制分为LLM文本流与仿真帧流两条独立通道。LLM文本流使用HTTP分块传输（Chunked Transfer Encoding），后端以NDJSON格式逐token返回{type, content, tool_call}结构，前端通过fetch().body.getReader()消费ReadableStream。仿真帧流采用SSE（Server-Sent Events），后端端点/api/sim/stream以固定频率（50ms/200ms/500ms，自适应）推送帧元数据，前端EventSource监听frame事件并触发帧图片的HTTP轮询。这两条流在用户发送消息时并发启动，通过共享的会话ID实现语义关联。'),
    create_body_para('仿真帧的写入与读取采用原子操作保证数据一致性。MCP仿真服务（PyBullet/Gazebo）将渲染帧写入共享目录（mcp/.sim_stream/latest.png），使用"写临时文件 + os.replace()"的原子替换模式，避免后端在写入过程中读到半帧数据。FastAPI直接通过文件系统服务PNG文件（/api/sim/frames），前端轮询间隔根据仿真状态动态调整：仿真运行中50ms轮询、空闲200ms、空闲超200步则降为500ms，降低空闲期后端负载。'),
]

for j, elem in enumerate(sections_454):
    body.insert(insert_idx + j, elem)
```

- [ ] **Step 2: 提交**

```bash
git commit -m "feat(doc): add 4.5.4 backend architecture and async streaming details"
```

---

## Task 7: 补全4.6 Training-Free GRPO Pipeline

**目标：** 找到"本实验设计了Training-free GRPO的pipeline。具体流程"这一段落，将其替换为完整的GRPO pipeline描述。

**新内容：**

> 4.6 Training-Free GRPO 经验强化机制
>
> 本节详细阐述Training-free GRPO的经验强化pipeline。该方法的核心思想在于通过轨迹比较而非模型参数更新来优化策略。系统每次任务执行中生成多条候选轨迹，并记录工具调用序列、关键中间状态及最终结果，然后利用外部Judge模型对同组轨迹进行相对打分，从而提炼出优势策略与失败经验。这些经验在不修改主模型参数的情况下，通过回灌至系统提示或代理上下文，实现策略持续改进。
>
> 经验轨迹收集阶段，系统对同一查询 q 先采样 G 条轨迹：τᵢ ~ πθ(·|q, Et), i = 1, …, G，其中 πθ 为冻结策略模型，Et 为第 t 轮经验库。每次attempt结束后，仿真环境自动重置（`CLEANUP_SIMULATION_PER_ATTEMPT = True`），保证轨迹间的独立性。轨迹提取三类信息：工具调用序列（MCP工具名与参数）、关键中间状态（仿真环境状态与观测结果）、最终结果（任务完成标志与输出文本）。
>
> 轨迹评分阶段由Judge LLM对每条轨迹独立评分，评价维度包括：任务完成质量（task_completion）、结果准确性（correctness）、输出清晰度（clarity）、异常处理质量（robustness）、简洁性（conciseness），以及综合得分（overall_score），各维度满分10分制。
>
> 经验对比与总结阶段，Judge Agent对高优势轨迹（S_top）与低优势轨迹（S_bottom）进行归因对比，生成结构化经验：summary（1-2句核心总结）、principles（3-5条核心原则）、dOs（应做事项，最多5条）、DON'Ts（应避免事项）。经验更新策略为：Et+1 = U(Et, Stop, Sbottom)。若新旧原则重叠≥3条则跳过，1-2条则部分更新，全新原则则写入新经验；每类Agent最多保留5条经验。
>
> 经验存储采用外部记忆库结构（`external_memory.json`），包含meta（方法名、时间戳）和experiences（经验列表，按agent_type分组）。生产环境中，`convert_experiences.py`将外部记忆过滤（评分阈值6.0）、分组后写入`agent_experiences.json`，backend启动时通过`_load_agent_experiences()`加载并注入各Agent系统提示：`P(t+1) = P_base ⊕ Format(E(t+1))`，形成"采样—评估—总结—回灌"的闭环优化机制。

**步骤：**

- [ ] **Step 1: 找到4.6 pipeline起始段落**

```python
body_children = list(body)
pipeline_start = None
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if '本实验设计了Training-free GRPO的pipeline' in text:
            pipeline_start = i
            print(f"Found pipeline start at {i}: {text[:60]}")
            break
```

- [ ] **Step 2: 找到4.6 pipeline段落结束位置（即第5章开始处）**

```python
chapter5_start = None
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if '5 实验设计' in text.strip() or '实验设计、结果与分析' in text:
            chapter5_start = i
            print(f"Chapter 5 starts at {i}: {text[:60]}")
            break
```

- [ ] **Step 3: 删除pipeline_start到chapter5_start之间的所有段落，插入新内容**

```python
# 删除旧内容
for i in range(chapter5_start - 1, pipeline_start - 1, -1):
    body.remove(body_children[i])

# 重新获取body_children
body_children = list(body)
# 找到4.6标题新位置
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if '4.6 Training-Free' in text.strip() and len(text.strip()) < 40:
            heading_idx = i
            break

# 插入新内容（在4.6标题后插入）
sections_46 = [
    create_body_para('本节详细阐述Training-free GRPO的经验强化pipeline。该方法的核心思想在于通过轨迹比较而非模型参数更新来优化策略。系统每次任务执行中生成多条候选轨迹，并记录工具调用序列、关键中间状态及最终结果，然后利用外部Judge模型对同组轨迹进行相对打分，从而提炼出优势策略与失败经验。这些经验在不修改主模型参数的情况下，通过回灌至系统提示或代理上下文，实现策略持续改进。'),
    create_body_para('经验轨迹收集阶段，系统对同一查询q先采样G条轨迹：τᵢ ~ πθ(·|q, Et), i = 1, …, G，其中πθ为冻结策略模型，Et为第t轮经验库。每次attempt结束后，仿真环境自动重置（CLEANUP_SIMULATION_PER_ATTEMPT = True），保证轨迹间的独立性。轨迹提取三类信息：工具调用序列（MCP工具名与参数）、关键中间状态（仿真环境状态与观测结果）、最终结果（任务完成标志与输出文本）。'),
    create_body_para('轨迹评分阶段由Judge LLM对每条轨迹独立评分，评价维度包括：任务完成质量（task_completion）、结果准确性（correctness）、输出清晰度（clarity）、异常处理质量（robustness）、简洁性（conciseness），以及综合得分（overall_score），各维度满分10分制。'),
    create_body_para('经验对比与总结阶段，Judge Agent对高优势轨迹（Stop）与低优势轨迹（Sbottom）进行归因对比，生成结构化经验：summary（1-2句核心总结）、principles（3-5条核心原则）、dOs（应做事项，最多5条）、DON'\''Ts（应避免事项）。经验更新策略为：Et+1 = U(Et, Stop, Sbottom)。若新旧原则重叠≥3条则跳过，1-2条则部分更新，全新原则则写入新经验；每类Agent最多保留5条经验。'),
    create_body_para('经验存储采用外部记忆库结构（external_memory.json），包含meta（方法名、时间戳）和experiences（经验列表，按agent_type分组）。生产环境中，convert_experiences.py将外部记忆过滤（评分阈值6.0）、分组后写入agent_experiences.json，backend启动时通过_load_agent_experiences()加载并注入各Agent系统提示：P(t+1) = P_base ⊕ Format(E(t+1))，形成"采样—评估—总结—回灌"的闭环优化机制。'),
]

for j, para in enumerate(sections_46):
    body.insert(heading_idx + 1 + j, para)
```

- [ ] **Step 4: 保存并最终验证**

```python
doc_xml_final = etree.tostring(root, xml_declaration=True, encoding='UTF-8', standalone=True)
file_contents['word/document.xml'] = doc_xml_final

with zipfile.ZipFile(OUTPUT_PATH, 'w', zipfile.ZIP_DEFLATED) as zout:
    for name, data in file_contents.items():
        zout.writestr(name, data)

print("Final save complete!")
```

验证：
```python
# 最终内容验证
body_children = list(body)
print("=== Final Chapter 4 verification ===")
for i, child in enumerate(body_children):
    if child.tag == f'{{{W_NS}}}p':
        texts = child.findall(f'.//{{{W_NS}}}t')
        text = ''.join([t.text or '' for t in texts])
        if ('4.5' in text or '4.6' in text or 'AuthView' in text or
            'ChatMainView' in text or 'Canvas' in text or 'async def' in text or
            'Training-free GRPO' in text or 'external_memory' in text or
            'τᵢ' in text):
            print(f"[{i}] {text[:100]}")
```

- [ ] **Step 5: 提交**

```bash
git add chapter4_filler.py
git commit -m "feat(doc): complete 4.6 GRPO pipeline and all Chapter 4 enhancements

- 4.5.1: Expand UI module with component details (AuthView, ChatMainView, ResultPanel)
- 4.5.2: Expand state management with concurrency and caching
- 4.5.3: Add frontend simulation rendering (Canvas 2D, RAF throttling, SSE reconnect)
- 4.5.4: Add backend architecture (async/await, SSE, NDJSON, atomic frame writes)
- 4.6: Complete Training-Free GRPO pipeline (collection, scoring, comparison, injection)"
```

---

## 自检清单

**Spec覆盖检查：**
- [x] 4.5.1 界面模块（AuthView、SessionSidebar、ChatMainView、ResultPanel）
- [x] 4.5.2 状态管理（多流并发、帧缓存、LRU、断线重连）
- [x] 4.5.3 前端渲染（Canvas 2D、requestAnimationFrame节流、SSE协调）
- [x] 4.5.4 后端架构（分层、async/await、流式响应、原子写入）
- [x] 4.6 GRPO完整pipeline（收集、评分、对比总结、存储注入）

**占位符扫描：**
- [x] 无"TBD"、"TODO"等占位符
- [x] 所有公式符号已用Unicode实际字符（τᵢ、πθ、Stop、Sbottom等）

**输出文件：** `2236127阮炜慈初稿_更新版.docx`
