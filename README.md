# 笔记分类器 (note-classifier) — GPU 分支

基于 `MoritzLaurer/bge-m3-zeroshot-v2.0` 的本地零样本分类器。**GPU 推理**（NVIDIA CUDA），将中文笔记按文段拆分为 9 个类别。

## 分类规则

- **单分类模式**：每段文本只归入一个分类文件，不重复归档。
- **决策逻辑**：
  1. 只考虑达到阈值（默认 0.40）的细分类别
  2. 分差 > 0.10 → 选最高分
  3. 分差 ≤ 0.10 → 按优先级选择
  4. 无类别达阈值 → `未分类.txt`
- **优先级**：疑问 > 待办 > 对话 > 身体 > 观点 > 想法 > 记录 > 驳论

## 分类体系

| 输出文件 | 细分类别 |
|----------|----------|
| 疑问.txt | 疑问、提问、待查询、待查 |
| 待办.txt | 待办、计划、任务、目标 |
| 对话.txt | 对话 |
| 身体.txt | 身体、健康、疾病、锻炼 |
| 观点.txt | 观点 |
| 想法.txt | 灵感、设计、点子、想法 |
| 记录.txt | 日记、美食、日常、反思 |
| 驳论.txt | 驳论 |
| 未分类.txt | 无类别达阈值 |

## 项目结构

```
note-classifier/
├── service.py          # 常驻分类服务（模型加载一次，持续监听队列）
├── client.py           # 客户端（拖文件即分类）
├── config.json         # 类别定义、阈值、优先级配置
├── requirements.txt    # Python 依赖
├── 笔记分类.bat         # 桌面入口（拖文件用）
├── 停止服务.bat         # 停止服务
├── cache/              # 模型缓存（自动生成）
├── queue/              # 队列目录（自动生成）
├── results/            # 完成标记（自动生成）
└── README.md
```

## 安装

```bash
pip install -r requirements.txt
```

首次运行会自动从 HuggingFace 下载模型（约 1 GB）到 `cache/` 目录。

### 环境要求

- Python 3.10+
- **NVIDIA GPU + CUDA**（需 torch CUDA 版）
- 约 2 GB 磁盘空间（模型 + 依赖）

## 使用

将 `.txt` 文件拖到 `笔记分类.bat` 上，或命令行运行：

```bash
python client.py file1.txt file2.txt
```

### 输出

结果输出到桌面：`{第一个文件名}_分类_{时间戳}/`

每个文段的输出格式：

```
[来源] 文件名
[文段 N]
[评分] 疑问=0.8146 | 提问=0.4186 | ...
[命中] 疑问、待查
[最终] 疑问.txt
[备注] 边界样本
原文内容...
```

### 性能

| 场景 | 耗时 |
|------|------|
| 冷启动（首次/超时后） | ~10 秒（含模型加载到 GPU） |
| 热运行（GPU 推理） | **~0.07 秒/段** |

服务空闲 5 分钟后自动退出释放显存。

## 服务管理

```bash
python service.py --status    # 查看状态
python service.py --stop      # 手动停止
python service.py             # 手动启动
```

## 配置

编辑 `config.json` 调整：

```json
{
  "threshold": 0.40,
  "score_margin": 0.10,
  "device": 0,
  "hypothesis_template": "This note is a {}",
  "priority": ["疑问", "待办", "对话", "身体", "观点", "想法", "记录", "驳论"]
}
```

## 技术细节

- 模型：`MoritzLaurer/bge-m3-zeroshot-v2.0`
- Pipeline：`transformers.zero-shot-classification`
- `multi_label=True`：各类别独立评分
- `hypothesis_template`：`"This note is a {}"`
- **GPU 推理**（`config.json` 中 `device: 0`；CPU 设为 `-1`）
- 长文段按句子边界分块后取平均
- 支持 UTF-8 / GB18030 / GBK / Big5 编码自动检测
- 文段分隔：连续两个及以上换行符
