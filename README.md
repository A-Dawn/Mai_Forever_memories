# Mai Forever Memories (定时聊天摘要与记忆插件)

> 嘿！或许，在麦麦陪伴你的日子里，你们一定有很多鲜活的回忆，也肯定想让她记住那份珍贵且真切的点点滴滴！现在，随着0.12版本的正式推出，lpmm知识库正式支持删除功能，“可塑性记忆”成为了现实！ ——ARC

## 启动说明（如何部署本插件）

以下说明只涉及插件的部署和启动前置条件

1. 插件配置（必须项）
   - 将插件中 `enabled`项目设置为 `true`
   - 确认插件配置文件 `plugins/mai_forever_memories/config.toml` 中 `plugin.enabled = true`
   - 确保全局 LPMM 开关已打开（否则摘要不会导入知识库）：
     - 在 `config/bot_config.toml` 中设置 `lpmm_knowledge.enable = true`
   - 如需性能告警功能，请在 `plugins/mai_forever_memories/config.toml` 中设置 `performance.admin_id` 为管理员的对话流ID（stream_id）（或 platform:ID 格式）。

2. 启动插件
   - 推荐方式（随主程序启动）
     - 启动整个 MaiBot 主程序，插件会在主程序启动时由插件管理器加载并在事件 `ON_START` 下启动调度器：
       - 在项目根目录运行主程序即可

3. 权限与路径
   - 确保 `paths.data_dir`（默认 `data/lpmm_summary`）目录具有写入权限


## 主要功能

- **定时自动摘要**：按照设定的时间点（每日/每周）自动提取聊天流中的关键信息。
- **多级记忆压缩**：
  - **每日摘要**：总结当天的对话。
  - **每周摘要**：将过去一周的每日摘要进一步压缩。
  - **二级压缩 (Level 2)**：当记忆过多时，将旧的每周摘要合并为更长期的背景知识。
- **永远的记忆 (Forever Memory)**：
  - **触发方式**：在聊天中对机器人说“记住这段”或“记住刚才”。
  - **功能**：立即分析最近 `lookback_messages` 条对话并生成独立摘要。
  - **特性**：该级别记忆（Level 3）被视为最高优先级，**永远不会被自动清理逻辑删除**。
- **LPMM 深度集成**：自动调用内部组件将摘要转化为知识图谱节点。
- **性能监控与安全防护**：
  - **任务前后检测**：自动检测 CPU 和内存占用。
  - **预警机制**：当资源占用超过阈值时，自动暂停后续任务并通知管理员。
  - **人工审核**：管理员可通过命令确认后恢复任务运行，确保系统稳定性。
- **智能分段处理**：针对 LPMM 优化，自动对 LLM 生成的文本进行分段预处理，提高知识提取质量。
- **灵活的流控制**：支持白名单/黑名单模式，可自由选择需要记录的群聊或私聊。
- **容量管理**：自动监控存储容量，防止数据无限膨胀。

## 配置项说明 (`config.toml`)

### [plugin]
- `enabled`: 是否启用插件。
- `config_version`: 配置文件版本。

### [schedule]
- `timezone`: 时区设置（如 `Asia/Shanghai` 或 `local`）。
- `daily_time`: 每日执行摘要的时间 (HH:MM)。
- `weekly_day`: 每周执行摘要的日期 (mon-sun)。
- `weekly_time`: 每周执行摘要的时间 (HH:MM)。
- `enable_weekly`: 是否启用每周汇总功能。

### [streams]
- `mode`: 过滤模式。`all` (全部), `allow` (仅白名单), `deny` (排除黑名单)。
- `allowlist`/`denylist`: 对应的流 ID（chat stream_id）列表，`allow` 模式下为空将不记录任何流。
- `include_group`/`include_private`: 是否包含群聊或私聊。

### [summary]
- `task`: 使用的 LLM 任务配置（默认为 `utils`）。
- `min_messages`: 触发摘要所需的最小消息数量，避免总结无意义的简短对话。
- `daily_max_chars`/`weekly_max_chars`: 不同级别摘要的目标长度。
- `filter_command`: 是否过滤掉以 `/` 开头的命令消息。

### [capacity]
- `max_paragraphs`: 允许存储的最大摘要段落数。
- `enable_level2`: 是否启用二级压缩（将旧摘要合并）。

### [paths]
- `data_dir`: 插件数据存储路径（默认为 `data/lpmm_summary`）。

### [performance]
- `admin_id`: 管理员标识（stream_id / platform:ID:private|group / 用户 ID）。
- `max_cpu_percent`: CPU 占用百分比阈值（默认 80）。
- `max_memory_percent`: 内存占用百分比阈值（默认 85）。
- `alert_interval`: 告警间隔（秒）。

## 管理命令

在聊天中使用以下命令手动管理插件：

- `/memories status`: 查看插件运行状态、下次运行时间及统计信息。
- `/memories list`: 列出当前聊天的所有摘要记录。
- `/memories show <ID>`: 查看特定摘要的详细内容。
- `/memories delete <ID>`: 删除特定摘要（注意：Level 3 记忆也可以被手动删除，但不会被自动清理）。
- `/memories approve`: 在收到性能预警后，由管理员发送此命令以恢复自动任务。
- `/memories run_daily`: 立即为当前聊天运行每日摘要任务。
- `/memories run_weekly`: 立即为当前聊天运行每周摘要任务。

## 注意事项

1. **全局开关**：本插件依赖于 MaiBot 的 LPMM 功能。请确保在全局配置 `config/bot_config.toml` 中设置了 `lpmm_knowledge.enable = true`，否则摘要将无法导入知识库。
2. **模型配置**：摘要生成依赖于 LLM。请确保您的 `model_config.toml` 中配置了有效的模型，并且 `utils` 任务（或您自定义的任务）可用。
3. **性能影响**：在执行每周摘要或大规模导入时，可能会占用一定的 CPU 和内存资源。建议将触发时间设置在麦麦休息的时间哦！
4. **数据安全**：摘要内容会存储在本地 `data/lpmm_summary` 目录下，请妥善保管。


## 消息反馈
如果在使用过程中出现问题/意见反馈，可通过提交issue/在技术群内联系ARC/发送邮件至`contact@luminarc.tech`进行反馈！
目前不接受PR,敬请谅解！