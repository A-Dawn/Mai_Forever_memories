import asyncio
import json
import os
import random
import re
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, TimeoutError as ProcessTimeoutError

try:
    import psutil
except ImportError:
    psutil = None

try:
    from zoneinfo import ZoneInfo
except Exception:  
    ZoneInfo = None

from src.config.config import global_config, model_config
# 为避免在主程序启动时立即导入并占用大量内存，
# 将可能引入大型数据/本地扩展的模块改为按需在函数内部导入。
from src.plugin_system import (
    BaseCommand,
    BaseEventHandler,
    BasePlugin,
    ComponentInfo,
    ConfigField,
    EventType,
    MaiMessages,
    BaseAction,
    ActionActivationType,
    chat_api,
    get_logger,
    llm_api,
    message_api,
    send_api,
    register_plugin,
)

logger = get_logger("mai_forever_memories")

_plugin_instance = None


DAY_SECONDS = 24 * 60 * 60
WEEK_SECONDS = 7 * DAY_SECONDS

# 容量管理相关常量
MAX_DELETE_BATCH_SIZE = 7  # 每次删除的最大条目数
MIN_WEEKLY_DAILY_ENTRIES = 7  # 每周摘要所需的最小每日摘要数
MAX_CAPACITY_ITERATIONS = 10  # 容量管理最大迭代次数


def _sanitize_filename(value: str) -> str:
    value = value or "unknown"
    return re.sub(r"[^A-Za-z0-9_.-]", "_", value)


def _now_ts() -> float:
    return time.time()


def _extract_message_content(message) -> str:
    """兼容多种消息对象，提取文本内容的通用方法。"""
    if not message:
        return ""
    # 常见字段优先级
    for key in ("content", "plain_text", "processed_plain_text", "raw_message", "message", "message_text"):
        try:
            val = getattr(message, key, None)
        except Exception:
            val = None
        if val:
            return str(val)

    # 支持 MaiMessages.message_segments（尝试拼接段文本）
    segs = getattr(message, "message_segments", None)
    if segs:
        parts = []
        try:
            for s in segs:
                p = getattr(s, "text", None) or getattr(s, "plain_text", None) or str(s)
                if p:
                    parts.append(str(p))
            if parts:
                return " ".join(parts)
        except Exception:
            pass

    return ""


class PendingConfirmTask:
    """待确认导入任务的数据类"""

    def __init__(self, task_id: str, task_type: str, chat_id: str, summary_text: str,
                 raw_path: Path, openie_path: Path, created_at: float):
        self.task_id = task_id
        self.task_type = task_type  # "daily", "weekly", "forever"
        self.chat_id = chat_id
        self.summary_text = summary_text
        self.raw_path = raw_path
        self.openie_path = openie_path
        self.created_at = created_at
        self.confirm_timeout = created_at + 300  # 默认5分钟超时

    def is_expired(self, current_time: float = None) -> bool:
        """检查任务是否已过期"""
        if current_time is None:
            current_time = time.time()
        return current_time >= self.confirm_timeout

    def to_dict(self) -> dict:
        """转换为字典格式，用于序列化"""
        return {
            "task_id": self.task_id,
            "task_type": self.task_type,
            "chat_id": self.chat_id,
            "summary_text": self.summary_text,
            "raw_path": str(self.raw_path),
            "openie_path": str(self.openie_path),
            "created_at": self.created_at,
            "confirm_timeout": self.confirm_timeout,
        }

    @classmethod
    def from_dict(cls, data: dict) -> 'PendingConfirmTask':
        """从字典格式创建实例"""
        return cls(
            task_id=data["task_id"],
            task_type=data["task_type"],
            chat_id=data["chat_id"],
            summary_text=data["summary_text"],
            raw_path=Path(data["raw_path"]),
            openie_path=Path(data["openie_path"]),
            created_at=data["created_at"],
        )


class MemoriesStartupHandler(BaseEventHandler):
    event_type = EventType.ON_START
    handler_name = "memories_startup"
    handler_description = "启动定时聊天摘要"

    async def execute(self, message: MaiMessages | None):
        if _plugin_instance:
            await _plugin_instance.start_scheduler()
        return True, True, None, None, None


class MemoriesStopHandler(BaseEventHandler):
    event_type = EventType.ON_STOP
    handler_name = "memories_stop"
    handler_description = "停止定时聊天摘要"

    async def execute(self, message: MaiMessages | None):
        if _plugin_instance:
            await _plugin_instance.stop_scheduler()
        return True, True, None, None, None


class MemoriesForeverHandler(BaseEventHandler):
    event_type = EventType.ON_MESSAGE
    handler_name = "memories_forever_trigger"
    handler_description = "自然语言触发永远的记忆"

    async def execute(self, message: MaiMessages | None):
        if not _plugin_instance or not message:
            return True, True, None, None, None
        # 兼容不同消息对象，提取文本内容
        content = _extract_message_content(message).strip()
        if not content:
            return True, True, None, None, None

        if not _plugin_instance.get_config("forever.enabled", True):
            return True, True, None, None, None
        
        # 如果启用了 Action 模式，则事件处理器跳过以避免与 Action 重复触发
        # 当用户希望用 Action 决策触发永远记忆时，应关闭事件触发器以防止双重执行
        if _plugin_instance.get_config("forever.use_action", True):
            return True, True, None, None, None
        
        keywords = _plugin_instance.get_config("forever.keywords", ["记住这段", "记住刚才"])
        
        if any(kw in content for kw in keywords):
            # 触发"永远的记忆"
            chat_id = message.chat_stream.stream_id if message.chat_stream else "unknown"
            task_name = f"forever_memory_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_forever(message),
                task_name=task_name
            )
            logger.info("已创建永远的记忆任务: %s", task_name)
            # 我们不拦截消息，让机器人继续处理（或者可以返回一个确认信息）
            return True, True, None, None, None
            
        return True, True, None, None, None


class MemoriesForeverAction(BaseAction):
    """
    Action 形式的 '永远的记忆'，通过平台的 Action 决策系统触发。
    """
    action_name = "memories_forever_action"
    action_description = "通过 Action 激活永远的记忆"
    activation_type = ActionActivationType.KEYWORD
    associated_types = ["text"]
    parallel_action = False
    action_parameters = {}
    # activation_keywords 将在 execute 时从插件配置读取，避免定义重复
    keyword_case_sensitive = False

    async def execute(self) -> tuple[bool, str]:
        """当 Action 被选中时调用，触发 run_forever 的后台任务。"""
        try:
            if not _plugin_instance or not _plugin_instance.get_config("forever.enabled", True):
                return False, "plugin disabled"

            # 检查是否启用了 action 模式
            if not _plugin_instance.get_config("forever.use_action", True):
                return False, "action mode disabled"

            # 获取消息内容
            content = None
            if getattr(self, "action_message", None):
                content = (
                    self.action_message.get("processed_plain_text")
                    or self.action_message.get("message")
                    or ""
                )
            # 兼容直接从 chat_stream 或其他字段读取
            if not content and getattr(self, "chat_stream", None) and getattr(self, "action_message", None) is None:
                content = ""

            if not content:
                return False, "no content"

            # 构造一个轻量的触发消息对象，供 run_forever 使用
            class _FakeTriggerMessage:
                def __init__(self, chat_stream, content, action_self):
                    self.chat_stream = chat_stream
                    self.content = content
                    self._action_self = action_self

                async def answer(self, text: str):
                    # 使用 action 实例提供的发送接口尝试回复（容错）
                    try:
                        await self._action_self.send_text(text, storage_message=False)
                    except Exception:
                        # 忽略发送错误
                        pass

            fm = _FakeTriggerMessage(getattr(self, "chat_stream", None), content, self)
            chat_id = fm.chat_stream.stream_id if fm.chat_stream and hasattr(fm.chat_stream, "stream_id") else "unknown"
            task_name = f"forever_action_{chat_id}"
            _plugin_instance._create_tracked_task(_plugin_instance.run_forever(fm), task_name=task_name)
            logger.info("Action 已创建永远的记忆任务: %s", task_name)
            return True, "scheduled"
        except Exception as exc:
            logger.error("MemoriesForeverAction 执行失败: %s", exc, exc_info=True)
            return False, "error"


class MemoriesCommand(BaseCommand):
    command_name = "memories"
    command_description = "管理定时聊天摘要"
    command_pattern = r"^/memories(?:\s+(?P<action>[a-zA-Z_]+))?(?:\s+(?P<arg>\S+))?$"

    async def execute(self):
        if not _plugin_instance:
            await self.send_text("记忆插件未就绪。", storage_message=False)
            return False, None, False
        if not self.get_config("commands.enabled", True):
            await self.send_text("记忆命令已禁用。", storage_message=False)
            return False, None, False

        action = ""
        arg = ""
        if self.matched_groups:
            action = (self.matched_groups.get("action", "") or "").lower().strip()
            arg = (self.matched_groups.get("arg", "") or "").strip()

        if action in ("", "status", "info"):
            stream_id = None
            if self.message.chat_stream and hasattr(self.message.chat_stream, "stream_id"):
                stream_id = self.message.chat_stream.stream_id
            status = await _plugin_instance.build_status_text(chat_id=stream_id)
            await self.send_text(status, storage_message=False)
            return True, None, False

        if action == "list":
            stream_id = None
            if self.message.chat_stream and hasattr(self.message.chat_stream, "stream_id"):
                stream_id = self.message.chat_stream.stream_id
            text = await _plugin_instance.get_recent_entries_text(chat_id=stream_id)
            await self.send_text(text, storage_message=False)
            return True, None, False

        if action == "show":
            if not arg:
                await self.send_text("请提供摘要 ID。用法: /memories show <ID>", storage_message=False)
                return False, None, False
            text = await _plugin_instance.get_entry_text(arg)
            await self.send_text(text, storage_message=False)
            return True, None, False

        if action == "delete":
            if not arg:
                await self.send_text("请提供摘要 ID。用法: /memories delete <ID>", storage_message=False)
                return False, None, False
            ok, msg = await _plugin_instance.manual_delete_entry(arg)
            await self.send_text(msg, storage_message=False)
            return ok, None, False

        if action == "approve":
            if not _plugin_instance._is_admin_message(self.message):
                await self.send_text("只有指定的管理员可以执行此操作。", storage_message=False)
                return False, None, False
            
            _plugin_instance._needs_approval_event.clear()  # 使用 Event 清除标志
            await self.send_text("性能警告已解除，摘要任务已恢复。", storage_message=False)
            return True, None, False

        if action in ("run_daily", "daily"):
            if not self.message.chat_stream:
                await self.send_text("缺少聊天流。", storage_message=False)
                return False, None, False
            chat_id = self.message.chat_stream.stream_id if self.message.chat_stream else "unknown"
            task_name = f"daily_summary_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_daily(streams=[self.message.chat_stream], manual=True),
                task_name=task_name
            )
            logger.info("已创建每日摘要任务: %s", task_name)
            await self.send_text("已为此聊天安排每日摘要任务。", storage_message=False)
            return True, None, False

        if action in ("run_weekly", "weekly"):
            if not self.message.chat_stream:
                await self.send_text("缺少聊天流。", storage_message=False)
                return False, None, False
            chat_id = self.message.chat_stream.stream_id if self.message.chat_stream else "unknown"
            task_name = f"weekly_summary_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_weekly(streams=[self.message.chat_stream], manual=True),
                task_name=task_name
            )
            logger.info("已创建每周摘要任务: %s", task_name)
            await self.send_text("已为此聊天安排每周摘要任务。", storage_message=False)
            return True, None, False

        if action == "now":
            if not self.message.chat_stream:
                await self.send_text("缺少聊天流。", storage_message=False)
                return False, None, False

            # 构造一个轻量的触发消息对象，供 run_forever 使用
            class _FakeCommandMessage:
                def __init__(self, chat_stream, command_self):
                    self.chat_stream = chat_stream
                    self._command_self = command_self

                async def answer(self, text: str):
                    # 使用命令实例提供的发送接口尝试回复（容错）
                    try:
                        await self._command_self.send_text(text, storage_message=False)
                    except Exception:
                        # 忽略发送错误
                        pass

            fm = _FakeCommandMessage(self.message.chat_stream, self)
            chat_id = fm.chat_stream.stream_id if fm.chat_stream and hasattr(fm.chat_stream, "stream_id") else "unknown"
            task_name = f"forever_memory_now_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_forever(fm),
                task_name=task_name
            )
            logger.info("已创建立即永久记忆任务: %s", task_name)
            await self.send_text("已为此聊天立即生成永久的记忆。", storage_message=False)
            return True, None, False

        if action in ("summarize_daily", "summarize-daily"):
            if not self.message.chat_stream:
                await self.send_text("缺少聊天流。", storage_message=False)
                return False, None, False
            chat_id = self.message.chat_stream.stream_id if self.message.chat_stream else "unknown"
            task_name = f"summarize_daily_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_daily(streams=[self.message.chat_stream], manual=True, import_flag=False),
                task_name=task_name
            )
            logger.info("已创建只总结每日摘要任务: %s", task_name)
            await self.send_text("已为此聊天安排只总结的每日摘要任务。", storage_message=False)
            return True, None, False

        if action in ("summarize_weekly", "summarize-weekly"):
            if not self.message.chat_stream:
                await self.send_text("缺少聊天流。", storage_message=False)
                return False, None, False
            chat_id = self.message.chat_stream.stream_id if self.message.chat_stream else "unknown"
            task_name = f"summarize_weekly_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_weekly(streams=[self.message.chat_stream], manual=True, import_flag=False),
                task_name=task_name
            )
            logger.info("已创建只总结每周摘要任务: %s", task_name)
            await self.send_text("已为此聊天安排只总结的每周摘要任务。", storage_message=False)
            return True, None, False

        if action in ("summarize_now", "summarize-now"):
            if not self.message.chat_stream:
                await self.send_text("缺少聊天流。", storage_message=False)
                return False, None, False

            # 构造一个轻量的触发消息对象，供 run_forever 使用
            class _FakeCommandMessage:
                def __init__(self, chat_stream, command_self):
                    self.chat_stream = chat_stream
                    self._command_self = command_self

                async def answer(self, text: str):
                    # 使用命令实例提供的发送接口尝试回复（容错）
                    try:
                        await self._command_self.send_text(text, storage_message=False)
                    except Exception:
                        # 忽略发送错误
                        pass

            fm = _FakeCommandMessage(self.message.chat_stream, self)
            chat_id = fm.chat_stream.stream_id if fm.chat_stream and hasattr(fm.chat_stream, "stream_id") else "unknown"
            task_name = f"summarize_now_{chat_id}"
            _plugin_instance._create_tracked_task(
                _plugin_instance.run_forever(fm, import_flag=False),
                task_name=task_name
            )
            logger.info("已创建只总结的即时记忆任务: %s", task_name)
            await self.send_text("已为此聊天立即生成只总结的记忆。", storage_message=False)
            return True, None, False

        if action == "confirm":
            if not _plugin_instance._is_admin_message(self.message):
                await self.send_text("只有指定的管理员可以执行此操作。", storage_message=False)
                return False, None, False

            # 解析参数：/memories confirm <task_id> <yes/no>
            parts = arg.split()
            if len(parts) != 2:
                await self.send_text("用法: /memories confirm <task_id> <yes/no>", storage_message=False)
                return False, None, False

            task_id, decision = parts
            decision = decision.lower()

            if decision not in ("yes", "no"):
                await self.send_text("决策必须是 'yes' 或 'no'。", storage_message=False)
                return False, None, False

            # 执行确认操作
            ok, msg = await _plugin_instance.handle_confirm_decision(task_id, decision == "yes")
            await self.send_text(msg, storage_message=False)
            return ok, None, False

        await self.send_text("用法: /memories [status|list|show|delete|daily|weekly|now|summarize-daily|summarize-weekly|summarize-now|confirm]", storage_message=False)
        return True, None, False


@register_plugin
class MaiForeverMemoriesPlugin(BasePlugin):
    """定时将聊天摘要导入 LPMM，支持保留和压缩。"""

    plugin_name: str = "mai_forever_memories"
    enable_plugin: bool = False
    dependencies: list[str] = []
    python_dependencies: list[str] = ["psutil"]
    config_file_name: str = "config.toml"

    config_section_descriptions = {
        "plugin": "插件设置",
        "schedule": "调度设置",
        "streams": "流选择",
        "summary": "摘要生成",
        "forever": "永远的记忆",
        "capacity": "保留与压缩",
        "paths": "存储路径",
        "performance": "性能检测",
        "commands": "命令设置",
    }

    config_schema: dict = {
        "plugin": {
            "config_version": ConfigField(type=str, default="1.0.0", description="配置版本"),
            "enabled": ConfigField(type=bool, default=False, description="启用插件"),
        },
        "schedule": {
            "timezone": ConfigField(
                type=str,
                default="Asia/Shanghai",
                description="IANA 时区或 'local'",
            ),
            "daily_time": ConfigField(type=str, default="11:30", description="每日触发时间 HH:MM"),
            "weekly_day": ConfigField(type=str, default="sun", description="每周触发日期 mon..sun"),
            "weekly_time": ConfigField(type=str, default="11:45", description="每周触发时间 HH:MM"),
            "enable_weekly": ConfigField(type=bool, default=True, description="启用每周摘要"),
        },
        "streams": {
            "mode": ConfigField(type=str, default="all", description="模式: all, allow, 或 deny"),
            "allowlist": ConfigField(type=list, default=[], description="允许的流 ID 列表"),
            "denylist": ConfigField(type=list, default=[], description="拒绝的流 ID 列表"),
            "include_group": ConfigField(type=bool, default=True, description="包含群聊"),
            "include_private": ConfigField(type=bool, default=True, description="包含私聊"),
        },
        "summary": {
            "task": ConfigField(type=str, default="utils", description="模型任务名称"),
            "min_messages": ConfigField(type=int, default=8, description="触发摘要的最小消息数"),
            "max_input_chars": ConfigField(type=int, default=8000, description="最大输入字符数"),
            "daily_max_chars": ConfigField(type=int, default=800, description="每日摘要长度"),
            "weekly_max_chars": ConfigField(type=int, default=1200, description="每周摘要长度"),
            "level2_max_chars": ConfigField(type=int, default=1600, description="最终压缩摘要长度"),
            "forever_max_chars": ConfigField(type=int, default=1000, description="永远的记忆长度"),
            "temperature": ConfigField(type=float, default=0.3, description="LLM 温度"),
            "filter_bot": ConfigField(type=bool, default=False, description="过滤机器人消息"),
            "filter_command": ConfigField(type=bool, default=True, description="过滤命令消息"),
            "truncate_messages": ConfigField(type=bool, default=True, description="截断长消息"),
            "auto_import": ConfigField(type=bool, default=True, description="自动导入摘要到知识库"),
            "confirm_import": ConfigField(type=bool, default=False, description="导入前是否需要管理员确认"),
            "confirm_stream": ConfigField(type=str, default="", description="发送确认消息的目标流ID，为空则使用performance.admin_id"),
            "confirm_timeout": ConfigField(type=int, default=300, description="确认超时时间（秒），超时后自动拒绝导入"),
        },
        "viewpoint": {
            "enabled": ConfigField(type=bool, default=False, description="启用观点总结功能"),
            "include_reply_style": ConfigField(type=bool, default=True, description="是否包含回复风格在主人设中"),
            "max_chars": ConfigField(type=int, default=500, description="观点总结最大长度"),
        },
        "forever": {
            "enabled": ConfigField(type=bool, default=True, description="启用自然语言触发"),
            "use_action": ConfigField(type=bool, default=True, description="启用 Action 模式触发永远的记忆"),
            "keywords": ConfigField(type=list, default=["记住这段", "记住刚才"], description="触发关键词"),
            "lookback_messages": ConfigField(type=int, default=20, description="回溯消息数"),
        },
        "capacity": {
            "max_paragraphs": ConfigField(type=int, default=200, description="最大存储摘要数"),
            "max_nodes": ConfigField(type=int, default=0, description="最大 KG 节点数 (0 禁用)"),
            "enable_level2": ConfigField(type=bool, default=True, description="允许最终压缩"),
        },
        "paths": {
            "data_dir": ConfigField(type=str, default="data/lpmm_summary", description="数据目录"),
        },
        "performance": {
            "enabled": ConfigField(type=bool, default=True, description="启用性能检测"),
            "max_cpu_percent": ConfigField(type=float, default=80.0, description="CPU 阈值"),
            "max_memory_percent": ConfigField(type=float, default=85.0, description="内存阈值"),
            "admin_id": ConfigField(
                type=str,
                default="",
                description="管理员标识: stream_id / platform:ID:private|group / 用户 ID",
            ),
            "alert_interval": ConfigField(type=int, default=3600, description="告警间隔"),
        },
        "commands": {
            "enabled": ConfigField(type=bool, default=True, description="启用 /memories 命令"),
        },
    }

    def __init__(self, *args, **kwargs):
        global _plugin_instance
        super().__init__(*args, **kwargs)
        self._stop_event = asyncio.Event()
        self._scheduler_task: asyncio.Task | None = None
        self._run_lock = asyncio.Lock()
        self._state_lock = asyncio.Lock()
        # 使用 Event 替代布尔变量，避免竞态条件
        self._needs_approval_event = asyncio.Event()
        self._last_alert_time = 0
        # 任务跟踪集合：用于跟踪所有后台任务，防止任务泄漏
        self._background_tasks: set[asyncio.Task] = set()
        self._tasks_lock = asyncio.Lock()
        # 限制并发的重型后台任务（如导入、索引重建）
        self._heavy_semaphore = asyncio.Semaphore(2)
        # 延迟创建进程池以避免在插件加载时立即启动子进程
        self._process_pool = None
        # 跟踪本插件按需导入的“重型”模块前缀，便于在任务完成后尝试卸载
        self._loaded_heavy_module_prefixes: set[str] = set()
        # KG 节点计数缓存（避免重复加载）
        self._cached_node_count: int | None = None
        self._node_count_cache_time: float = 0.0
        self._node_count_cache_ttl: float = 60.0  # 缓存有效期60秒
        plugin_dir = Path(self.plugin_dir).resolve()
        self._root_dir = plugin_dir.parent.parent
        self._data_dir = self._resolve_data_dir(self.get_config("paths.data_dir", "data/lpmm_summary"))
        self._raw_dir = self._data_dir / "raw"
        self._openie_dir = self._data_dir / "openie"
        self._delete_dir = self._data_dir / "delete"
        self._state_path = self._data_dir / "state.json"
        # 待确认导入任务管理
        self._pending_confirm_tasks: dict[str, dict] = {}
        self._pending_tasks_lock = asyncio.Lock()
        _plugin_instance = self

    def get_plugin_components(self) -> list[tuple[ComponentInfo, type]]:
        components = [
            (MemoriesStartupHandler.get_handler_info(), MemoriesStartupHandler),
            (MemoriesStopHandler.get_handler_info(), MemoriesStopHandler),
            (MemoriesForeverHandler.get_handler_info(), MemoriesForeverHandler),
        ]
        # 如果启用了 action 模式，则注册 MemoriesForeverAction
        try:
            if self.get_config("forever.use_action", True):
                components.append((MemoriesForeverAction.get_action_info(), MemoriesForeverAction))
        except Exception:
            # 容错：如果插件加载环境不支持 Action 类型或 get_action_info，不影响命令/事件处理
            logger.debug("未能注册 MemoriesForeverAction，可能是宿主不支持 Action：%s", sys.exc_info()[0])
        if self.config.get("commands", {}).get("enabled", True):
            components.append((MemoriesCommand.get_command_info(), MemoriesCommand))
        return components

    def _create_tracked_task(self, coro, task_name: str = "unknown") -> asyncio.Task:
        """创建并跟踪后台任务，自动清理已完成的任务并记录异常。"""
        async def _task_wrapper():
            try:
                logger.debug("任务开始执行: %s", task_name)
                return await coro
            except asyncio.CancelledError:
                logger.info("任务被取消: %s", task_name)
                raise
            except Exception as exc:
                logger.error("任务执行失败: %s, 错误: %s", task_name, exc, exc_info=True)
                raise

        task = asyncio.create_task(_task_wrapper())
        
        # 添加完成回调，用于清理任务跟踪
        def _on_task_done(t: asyncio.Task):
            async def _cleanup():
                async with self._tasks_lock:
                    if t in self._background_tasks:
                        self._background_tasks.discard(t)
                        logger.debug("任务已从跟踪集合移除: %s (剩余: %d)", task_name, len(self._background_tasks))
                    # 检查是否有异常
                    if t.exception():
                        logger.warning("任务 %s 完成时包含异常: %s", task_name, t.exception())
            
            # 如果事件循环正在运行，创建清理任务；否则直接执行（很少见的情况）
            try:
                loop = asyncio.get_running_loop()
                if loop.is_running():
                    cleanup_task = asyncio.create_task(_cleanup())
                    # 添加异常处理，防止清理任务失败被静默吞没
                    cleanup_task.add_done_callback(
                        lambda t: logger.error("清理任务失败: %s", t.exception(), exc_info=True)
                        if t.exception() else None
                    )
                else:
                    # 这种情况应该很少见，但为了安全起见
                    import warnings
                    warnings.warn(f"在非运行的事件循环中清理任务: {task_name}")
            except RuntimeError:
                # 没有运行的事件循环，这种情况应该很少见
                logger.warning("无法获取事件循环，任务清理可能失败: %s", task_name)
        
        task.add_done_callback(_on_task_done)
        
        # 将任务添加到跟踪集合
        async def _add_task():
            async with self._tasks_lock:
                self._background_tasks.add(task)
                logger.debug("任务已添加到跟踪集合: %s (总数: %d)", task_name, len(self._background_tasks))
        
        # 如果当前在事件循环中，直接添加
        try:
            loop = asyncio.get_running_loop()
            if loop.is_running():
                # 创建一个立即执行的协程来添加任务
                add_task_task = asyncio.create_task(_add_task())
                # 添加异常处理，防止添加任务失败被静默吞没
                add_task_task.add_done_callback(
                    lambda t: logger.error("添加任务到跟踪集合失败: %s", t.exception(), exc_info=True)
                    if t.exception() else None
                )
            else:
                # 如果不在运行的事件循环中，同步添加（这种情况应该很少）
                import warnings
                warnings.warn(f"在非运行的事件循环中添加任务: {task_name}")
        except RuntimeError:
            # 没有运行的事件循环，这种情况应该很少见
            logger.warning("无法获取事件循环，任务可能无法正确跟踪: %s", task_name)
        
        return task


    async def _cleanup_finished_tasks(self) -> None:
        """清理已完成的任务（定期调用）。"""
        async with self._tasks_lock:
            finished_tasks = [t for t in self._background_tasks if t.done()]
            for task in finished_tasks:
                self._background_tasks.discard(task)
                # 检查是否有异常
                if task.exception():
                    logger.warning("发现已完成的任务包含异常: %s", task.exception())
            if finished_tasks:
                logger.debug("清理了 %d 个已完成的任务 (剩余: %d)", len(finished_tasks), len(self._background_tasks))

    def get_active_tasks_count(self) -> int:
        """获取当前活跃的后台任务数量。"""
        return len(self._background_tasks)

    def _resolve_data_dir(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self._root_dir / path
        return path

    def _check_performance(self) -> tuple[bool, str]:
        """检查 CPU 和内存占用，返回 (是否危险, 描述信息)。"""
        if not psutil or not self.get_config("performance.enabled", True):
            return False, ""
        
        cpu_usage = psutil.cpu_percent(interval=0.5)
        mem_usage = psutil.virtual_memory().percent
        
        max_cpu = float(self.get_config("performance.max_cpu_percent", 80.0))
        max_mem = float(self.get_config("performance.max_memory_percent", 85.0))
        
        is_dangerous = cpu_usage > max_cpu or mem_usage > max_mem
        desc = f"当前性能: CPU {cpu_usage}% (阈值 {max_cpu}%), 内存 {mem_usage}% (阈值 {max_mem}%)"
        
        if is_dangerous:
            self._needs_approval_event.set()  # 使用 Event 设置标志
            logger.warning("性能预警: %s", desc)
        
        return is_dangerous, desc



    async def _alert_admin(self, message: str) -> None:
        """向管理员发送性能告警。"""
        admin_id = self.get_config("performance.admin_id", "")
        if not admin_id:
            return

        now = time.time()
        interval = int(self.get_config("performance.alert_interval", 3600))
        if now - self._last_alert_time < interval:
            return

        self._last_alert_time = now
        admin_stream_id, _ = self._resolve_admin_target(str(admin_id))
        if not admin_stream_id:
            logger.warning("未找到管理员的聊天流，无法发送性能告警。")
            return
        await send_api.text_to_stream(
            f"⚠️ 【记忆插件性能告警】\n{message}\n\n系统已进入保护模式，后续摘要任务需要您发送 `/memories approve` 确认后方可继续执行。",
            admin_stream_id,
            storage_message=False,
        )

    def _resolve_admin_target(self, admin_id: str) -> tuple[str | None, str | None]:
        admin_id = (admin_id or "").strip()
        if not admin_id:
            return None, None
        parts = admin_id.split(":")
        if len(parts) == 3 and parts[2] in ("private", "group"):
            platform, target_id, target_type = parts
            stream = None
            if target_type == "private":
                stream = chat_api.get_stream_by_user_id(target_id, platform)
            else:
                stream = chat_api.get_stream_by_group_id(target_id, platform)
            return (stream.stream_id if stream else None), None
        if re.fullmatch(r"[a-fA-F0-9]{32}", admin_id):
            return admin_id.lower(), None
        stream_id = self._find_stream_id_by_user_id(admin_id)
        return stream_id, admin_id

    def _find_stream_id_by_user_id(self, user_id: str) -> str | None:
        if not user_id:
            return None
        for stream in chat_api.get_private_streams(chat_api.SpecialTypes.ALL_PLATFORMS):
            info = getattr(stream, "user_info", None)
            if info and str(info.user_id) == str(user_id):
                return stream.stream_id
        return None

    def _get_message_user_id(self, message) -> str:
        info = getattr(message, "message_info", None)
        user_info = getattr(info, "user_info", None)
        user_id = getattr(user_info, "user_id", "")
        return str(user_id) if user_id is not None else ""

    def _is_admin_message(self, message) -> bool:
        admin_id = str(self.get_config("performance.admin_id", "") or "").strip()
        if not admin_id:
            return True
        if not message:
            return False
        stream_id = None
        if getattr(message, "chat_stream", None) and hasattr(message.chat_stream, "stream_id"):
            stream_id = message.chat_stream.stream_id
        user_id = self._get_message_user_id(message)
        admin_stream_id, admin_user_id = self._resolve_admin_target(admin_id)
        if admin_stream_id and stream_id and stream_id == admin_stream_id:
            return True
        if admin_user_id and user_id and user_id == admin_user_id:
            return True
        return False

    async def _precheck_performance(self, manual: bool) -> bool:
        is_dangerous, desc = self._check_performance()
        if is_dangerous:
            await self._alert_admin(desc)
            if not manual:
                return False
        return True

    def _ensure_dirs(self) -> None:
        for path in [self._data_dir, self._raw_dir, self._openie_dir, self._delete_dir]:
            path.mkdir(parents=True, exist_ok=True)

    def _is_enabled(self) -> bool:
        return bool(self.get_config("plugin.enabled", False))

    def _lpmm_enabled(self) -> bool:
        return bool(global_config.lpmm_knowledge.enable)

    async def start_scheduler(self) -> None:
        if not self._is_enabled():
            logger.info("记忆插件已禁用；调度器未启动。")
            return
        if self._scheduler_task and not self._scheduler_task.done():
            return
        self._ensure_dirs()
        self._stop_event.clear()
        self._scheduler_task = asyncio.create_task(self._scheduler_loop())
        logger.info("记忆调度器已启动。")

    async def stop_scheduler(self) -> None:
        if not self._scheduler_task:
            return
        self._stop_event.set()
        if not self._scheduler_task.done():
            self._scheduler_task.cancel()
            try:
                await self._scheduler_task
            except asyncio.CancelledError:
                pass
        self._scheduler_task = None
        
        # 取消所有后台任务
        async with self._tasks_lock:
            active_tasks = list(self._background_tasks)
            if active_tasks:
                logger.info("正在取消 %d 个后台任务...", len(active_tasks))
                for task in active_tasks:
                    if not task.done():
                        task.cancel()
                # 等待所有任务完成或取消
                if active_tasks:
                    await asyncio.gather(*active_tasks, return_exceptions=True)
                self._background_tasks.clear()
                logger.info("所有后台任务已清理")
        # 关闭进程池（如果存在），避免子进程遗留
        try:
            if getattr(self, "_process_pool", None):
                try:
                    self._process_pool.shutdown(wait=False)
                except Exception:
                    pass
        except Exception:
            pass
        
        logger.info("记忆调度器已停止。")

    async def _scheduler_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                await self._run_due()
                # 定期清理已完成的任务
                await self._cleanup_finished_tasks()
                # 清理过期的待确认任务
                await self._cleanup_expired_confirm_tasks()
                
                next_run = self._next_scheduled_run()
                if next_run is None:
                    await asyncio.wait_for(self._stop_event.wait(), timeout=3600)
                    continue
                now = datetime.now(self._get_timezone())
                wait_sec = max(1.0, (next_run - now).total_seconds())
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_sec)
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                break
            except Exception as exc:
                logger.error("调度器错误: %s", exc, exc_info=True)
                await asyncio.sleep(5)

    def _get_timezone(self) -> datetime.tzinfo:
        tz_name = str(self.get_config("schedule.timezone", "local"))
        if tz_name.lower() in ("local", "system"):
            return datetime.now().astimezone().tzinfo
        if ZoneInfo is None:
            return datetime.now().astimezone().tzinfo
        try:
            return ZoneInfo(tz_name)
        except Exception:
            logger.warning("无效的时区 '%s'，回退到本地时区。", tz_name)
            return datetime.now().astimezone().tzinfo

    def _parse_time_str(self, value: str, fallback: str) -> tuple[int, int]:
        value = (value or fallback).strip()
        match = re.match(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$", value)
        if not match:
            value = fallback
            match = re.match(r"^(?P<h>\d{1,2}):(?P<m>\d{2})$", value)
        if not match:
            return 11, 30
        hour = int(match.group("h"))
        minute = int(match.group("m"))
        if hour < 0 or hour > 23 or minute < 0 or minute > 59:
            return 11, 30
        return hour, minute

    def _parse_weekday(self, value: str) -> int:
        mapping = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        key = (value or "sun").strip().lower()
        return mapping.get(key, 6)

    def _daily_anchor(self, now: datetime) -> datetime:
        hour, minute = self._parse_time_str(
            str(self.get_config("schedule.daily_time", "11:30")), "11:30"
        )
        anchor = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now >= anchor:
            return anchor
        return anchor - timedelta(days=1)

    def _weekly_anchor(self, now: datetime) -> datetime:
        hour, minute = self._parse_time_str(
            str(self.get_config("schedule.weekly_time", "11:45")), "11:45"
        )
        target = self._parse_weekday(str(self.get_config("schedule.weekly_day", "sun")))
        anchor = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        delta = (anchor.weekday() - target) % 7
        anchor = anchor - timedelta(days=delta)
        if now >= anchor:
            return anchor
        return anchor - timedelta(days=7)

    def _next_daily_time(self, now: datetime) -> datetime:
        hour, minute = self._parse_time_str(
            str(self.get_config("schedule.daily_time", "11:30")), "11:30"
        )
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        if now < target:
            return target
        return target + timedelta(days=1)

    def _next_weekly_time(self, now: datetime) -> datetime:
        hour, minute = self._parse_time_str(
            str(self.get_config("schedule.weekly_time", "11:45")), "11:45"
        )
        target_day = self._parse_weekday(str(self.get_config("schedule.weekly_day", "sun")))
        target = now.replace(hour=hour, minute=minute, second=0, microsecond=0)
        delta = (target_day - target.weekday()) % 7
        if delta == 0 and now >= target:
            delta = 7
        target = target + timedelta(days=delta)
        return target

    def _next_scheduled_run(self) -> datetime | None:
        tz = self._get_timezone()
        now = datetime.now(tz)
        daily = self._next_daily_time(now)
        weekly = None
        if bool(self.get_config("schedule.enable_weekly", True)):
            weekly = self._next_weekly_time(now)
        if weekly is None:
            return daily
        return daily if daily <= weekly else weekly

    async def build_status_text(self, chat_id: str | None = None) -> str:
        # 在异步上下文中调用此方法时请使用 await build_status_text(...)
        # 使用线程执行磁盘 I/O 以避免阻塞事件循环
        state = await asyncio.to_thread(self._load_state_sync)
        tz = self._get_timezone()
        now = datetime.now(tz)
        next_daily = self._next_daily_time(now)
        next_weekly = None
        if bool(self.get_config("schedule.enable_weekly", True)):
            next_weekly = self._next_weekly_time(now)
        active_entries = [e for e in state["entries"] if not e.get("deleted")]
        daily_count = sum(1 for e in active_entries if e.get("level") == 0)
        weekly_count = sum(1 for e in active_entries if e.get("level") == 1)
        level2_count = sum(1 for e in active_entries if e.get("level") == 2)
        forever_count = sum(1 for e in active_entries if e.get("level") == 3)
        tz_name = getattr(tz, "tzname", lambda _=None: str(tz))(None)
        active_tasks = self.get_active_tasks_count()
        lines = [
            "Mai Forever Memories 状态",
            f"已启用: {self._is_enabled()}",
            f"LPMM 已启用: {self._lpmm_enabled()}",
            f"时区: {tz_name}",
            f"每日时间: {self.get_config('schedule.daily_time', '11:30')}",
            f"每周时间: {self.get_config('schedule.weekly_time', '11:45')}",
            f"下次每日: {next_daily.isoformat()}",
            f"下次每周: {next_weekly.isoformat() if next_weekly else '已禁用'}",
            f"条目数: 每日={daily_count} 每周={weekly_count} 二级={level2_count} 永远={forever_count}",
            f"活跃后台任务: {active_tasks}",
        ]
        if chat_id:
            last_daily = float(state["last_daily_run"].get(chat_id, 0) or 0)
            last_weekly = float(state["last_weekly_run"].get(chat_id, 0) or 0)
            lines.append(
                f"上次每日运行: {datetime.fromtimestamp(last_daily, tz).isoformat() if last_daily else '-'}"
            )
            lines.append(
                f"上次每周运行: {datetime.fromtimestamp(last_weekly, tz).isoformat() if last_weekly else '-'}"
            )
        return "\n".join(lines)

    async def get_recent_entries_text(self, chat_id: str | None = None, limit: int = 10) -> str:
        state = await asyncio.to_thread(self._load_state_sync)
        tz = self._get_timezone()
        entries = [e for e in state["entries"] if not e.get("deleted")]
        if chat_id:
            entries = [e for e in entries if e.get("chat_id") == chat_id]
        
        entries = sorted(entries, key=lambda e: e.get("created_at", 0), reverse=True)[:limit]
        if not entries:
            return "未找到摘要条目。"
        
        lines = ["最近的摘要条目:"]
        for e in entries:
            created_at = float(e.get("created_at", 0) or 0)
            dt = datetime.fromtimestamp(created_at, tz)
            kind = e.get("kind", "unknown")
            eid = e.get("id", "unknown")
            lines.append(f"- [{dt.strftime('%Y-%m-%d %H:%M')}] {kind.upper()}: {eid}")
        
        lines.append("\n使用 /memories show <ID> 查看详情。")
        return "\n".join(lines)

    async def get_entry_text(self, entry_id: str) -> str:
        state = await asyncio.to_thread(self._load_state_sync)
        entry = next((e for e in state["entries"] if e.get("id") == entry_id), None)
        if not entry:
            return f"未找到 ID 为 {entry_id} 的条目。"
        
        if entry.get("deleted"):
            return f"条目 {entry_id} 已被删除。"
        
        raw_file = entry.get("raw_file")
        if not raw_file:
            return "条目缺少原始文件路径。"
        
        path = self._resolve_entry_path(raw_file)
        if not path.exists():
            return f"原始文件不存在: {raw_file}"
        
        try:
            content = await asyncio.to_thread(path.read_text, "utf-8")
            return f"摘要详情 ({entry_id}):\n\n{content}"
        except Exception as exc:
            return f"读取文件失败: {exc}"

    async def manual_delete_entry(self, entry_id: str) -> tuple[bool, str]:
        state = await self._load_state()
        entry = next((e for e in state["entries"] if e.get("id") == entry_id), None)
        if not entry:
            return False, f"未找到 ID 为 {entry_id} 的条目。"
        
        if entry.get("deleted"):
            return True, f"条目 {entry_id} 已经处于删除状态。"
        
        ok = await self._delete_entries([entry])
        if ok:
            await self._save_state(state)
            return True, f"成功删除条目 {entry_id} 及其关联的 LPMM 知识。"
        else:
            return False, f"删除条目 {entry_id} 失败，请检查日志。"

    def _select_streams(self):
        streams = []
        include_group = bool(self.get_config("streams.include_group", True))
        include_private = bool(self.get_config("streams.include_private", True))
        if include_group:
            streams.extend(chat_api.get_group_streams(chat_api.SpecialTypes.ALL_PLATFORMS))
        if include_private:
            streams.extend(chat_api.get_private_streams(chat_api.SpecialTypes.ALL_PLATFORMS))

        mode = str(self.get_config("streams.mode", "all")).lower().strip()
        allowlist = set(self.get_config("streams.allowlist", []) or [])
        denylist = set(self.get_config("streams.denylist", []) or [])
        filtered = []
        for stream in streams:
            stream_id = getattr(stream, "stream_id", None)
            if not stream_id:
                continue
            if mode == "allow":
                if stream_id in allowlist:
                    filtered.append(stream)
            elif mode == "deny":
                if stream_id not in denylist:
                    filtered.append(stream)
            else:
                filtered.append(stream)
        return filtered

    def _stream_allowed(self, stream) -> bool:
        if not stream:
            return False
        stream_id = getattr(stream, "stream_id", None)
        if not stream_id:
            return False
        mode = str(self.get_config("streams.mode", "all")).lower().strip()
        allowlist = set(self.get_config("streams.allowlist", []) or [])
        denylist = set(self.get_config("streams.denylist", []) or [])
        if mode == "allow":
            return stream_id in allowlist
        if mode == "deny":
            return stream_id not in denylist
        return True

    def _default_state(self) -> dict:
        return {
            "version": 1,
            "last_daily_run": {},
            "last_weekly_run": {},
            "finalized_level2": {},
            "entries": [],
        }

    def _load_state_sync(self) -> dict:
        """同步加载状态文件（只读操作，不保证并发安全）。"""
        if not self._state_path.exists():
            return self._default_state()
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            logger.error("状态文件 JSON 解析失败: %s, 使用默认状态", exc)
            return self._default_state()
        except Exception as exc:
            logger.error("加载状态文件失败: %s, 使用默认状态", exc, exc_info=True)
            return self._default_state()
        if not isinstance(data, dict):
            logger.warning("状态文件格式无效，使用默认状态")
            return self._default_state()
        data.setdefault("version", 1)
        data.setdefault("last_daily_run", {})
        data.setdefault("last_weekly_run", {})
        data.setdefault("finalized_level2", {})
        data.setdefault("entries", [])
        if not isinstance(data["entries"], list):
            logger.warning("状态文件 entries 字段格式无效，重置为空列表")
            data["entries"] = []
        return data

    async def _load_state(self) -> dict:
        """加载状态文件，使用锁保护并发访问。"""
        async with self._state_lock:
            return self._load_state_sync()

    async def _save_state(self, state: dict) -> None:
        """保存状态文件，使用锁保护并发访问。"""
        async with self._state_lock:
            self._ensure_dirs()
            tmp_path = self._state_path.with_suffix(".tmp")
            try:
                tmp_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(tmp_path, self._state_path)
            except Exception as exc:
                logger.error("保存状态文件失败: %s", exc, exc_info=True)
                # 清理临时文件
                if tmp_path.exists():
                    try:
                        tmp_path.unlink()
                    except Exception as cleanup_exc:
                        logger.warning("清理临时状态文件失败: %s", cleanup_exc)
                raise

    def _relpath(self, path: Path) -> str:
        try:
            return str(path.relative_to(self._root_dir))
        except Exception:
            return str(path)

    def _resolve_entry_path(self, value: str) -> Path:
        path = Path(value)
        if not path.is_absolute():
            path = self._root_dir / path
        return path

    def _get_task_config(self):
        task_name = str(self.get_config("summary.task", "utils"))
        try:
            return model_config.model_task_config.get_task(task_name)
        except Exception:
            logger.warning("未知的任务配置 '%s'，使用 utils。", task_name)
            return model_config.model_task_config.utils

    def _truncate_text(self, text: str, max_chars: int) -> str:
        if max_chars and len(text) > max_chars:
            return text[:max_chars].rstrip()
        return text

    def _preprocess_chat_content(self, content: str) -> str:
        """预处理聊天内容，过滤掉可能的插件回复和系统消息。

        过滤规则：
        1. 过滤包含大量表情符号的消息（通常是插件输出）
        2. 过滤格式化的状态报告消息
        3. 过滤固定的确认/错误消息模式
        4. 过滤以特定关键词开头的消息
        """
        if not content:
            return content

        lines = content.split('\n')
        filtered_lines = []

        # 过滤规则
        plugin_indicators = [
            # 大量表情符号（通常是插件状态报告）
            lambda line: sum(1 for c in line if ord(c) > 0x1F600 and ord(c) < 0x1F64F) > 2,  # 表情符号
            lambda line: sum(1 for c in line if ord(c) > 0x1F300 and ord(c) < 0x1F5FF) > 2,  # 符号
            lambda line: sum(1 for c in line if ord(c) > 0x1F680 and ord(c) < 0x1F6FF) > 2,  # 交通符号
            # 格式化的状态报告
            lambda line: '❌' in line and ('未找到' in line or '失败' in line or '错误' in line),
            lambda line: '✅' in line and ('成功' in line or '完成' in line),
            lambda line: '📋' in line and ('显示' in line or '列表' in line),
            lambda line: '⏰' in line and '时间' in line,
            # 插件确认消息
            lambda line: line.startswith('好的') and ('记忆' in line or '任务' in line),
            lambda line: '已为此聊天' in line and ('安排' in line or '生成' in line),
            lambda line: '抱歉' in line and '错误' in line,
            # 命令执行结果
            lambda line: any(line.startswith(prefix) for prefix in ['用法:', '请提供', '只有', '缺少']),
        ]

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # 检查是否匹配任何插件指示器
            is_plugin_output = any(indicator(line) for indicator in plugin_indicators)

            if not is_plugin_output:
                filtered_lines.append(line)

        return '\n'.join(filtered_lines)

    def _build_prompt(self, title: str, max_chars: int, content: str) -> str:
        return (
            "你是一个专业的聊天记录摘要生成器，专门为LPMM知识库生成高质量内容。\n"
            f"任务: {title}\n"
            f"限制: 最大 {max_chars} 字符。\n"
            "核心要求:\n"
            "1. 重点关注事实、决策、任务和后续行动，保持极高的信息密度。\n"
            "2. 必须进行分段处理：将不同的讨论主题或事件分成独立的段落。\n"
            "3. 段落之间必须使用一个完整的空行分隔，以便于知识库提取。\n"
            "4. 每个段落应语义完整，避免零散的句子。\n"
            "5. 仅输出摘要纯文本，不要包含任何引言或解释。\n"
            "6. 忽略所有命令消息（如以/开头的消息）、插件回复、系统通知、状态报告、确认消息和错误提示。\n"
            "7. 只总结真正的聊天对话内容，避免总结机器人或其他插件的输出信息。\n"
            "质量要求:\n"
            "8. 确保内容表述的一致性，避免同一事件出现不同表述方式。\n"
            "9. 明确标注关键时间节点和事件发生的具体时间点。\n"
            "10. 强化实体间关系的表达，明确说明'谁对谁做了什么'或'谁与谁的关系如何'。\n"
            "11. 使用具体的实体名称，避免模糊指代，使用'具体人名'而非'某人'或'成员'。\n"
            "12. 突出重要决策、达成共识和后续行动计划，这些是知识库最有价值的信息。\n"
            "13. 对于固定的事实内容（如歌词、特定短语、诗句等），使用特殊标记保留原始格式：\n"
            "    [ORIGINAL:标签]原始内容[/ORIGINAL]\n"
            "    例如：[ORIGINAL:副歌]天空之城，永远的传说[/ORIGINAL]\n\n"
            "内容:\n"
            f"{content}\n"
        )

    def _preprocess_summary_for_lpmm(self, text: str) -> str:
        """对摘要进行预处理，确保符合 LPMM 的分段要求。"""
        # 统一换行符并去除多余空白
        lines = [line.strip() for line in text.splitlines()]

        # 特殊处理：保留原始内容的标记
        paragraphs = []
        current_para = []
        in_original_block = False

        for line in lines:
            if line.startswith('[ORIGINAL:'):
                # 开始原始内容块，保持原格式
                if current_para:
                    paragraphs.append(" ".join(current_para))
                    current_para = []
                in_original_block = True
                # 保留原始标记
                paragraphs.append(line)
            elif line.startswith('[/ORIGINAL]'):
                # 结束原始内容块
                in_original_block = False
                paragraphs.append(line)
            elif in_original_block:
                # 在原始内容块内，保持原始格式
                paragraphs.append(line)
            elif line:
                current_para.append(line)
            elif current_para:
                paragraphs.append(" ".join(current_para))
                current_para = []

        if current_para:
            paragraphs.append(" ".join(current_para))

        return "\n\n".join(paragraphs)

    def _format_lyrics_with_original_preservation(self, lyrics_data: dict) -> str:
        """专门处理歌词内容，保留原始格式的同时添加结构化描述"""
        formatted_parts = []

        # 歌曲基本信息
        if 'title' in lyrics_data:
            formatted_parts.append(f"歌曲《{lyrics_data['title']}》")

        if 'artist' in lyrics_data:
            formatted_parts.append(f"由{lyrics_data['artist']}演唱")

        if 'genre' in lyrics_data:
            formatted_parts.append(f"是一首{lyrics_data['genre']}风格歌曲")

        formatted_parts.append("")  # 空行分隔

        # 处理歌词结构
        if 'structure' in lyrics_data:
            for section in lyrics_data['structure']:
                section_type = section.get('type', 'lyrics')
                content = section.get('content', '')

                if section_type == 'original':
                    # 保留原始歌词格式
                    formatted_parts.append(f"[ORIGINAL:{section.get('label', '歌词')}]")
                    # 保持原始断句和格式
                    formatted_parts.append(content)
                    formatted_parts.append("[/ORIGINAL]")
                else:
                    # 结构化描述
                    formatted_parts.append(f"[{section_type.upper()}]")
                    formatted_parts.append(content)

                formatted_parts.append("")  # 段落分隔

        # 整体分析
        if 'analysis' in lyrics_data:
            formatted_parts.append("整体分析：")
            for key, value in lyrics_data['analysis'].items():
                formatted_parts.append(f"{key}：{value}")
            formatted_parts.append("")  # 空行分隔

        return "\n\n".join(formatted_parts)

    def _extract_entities_from_lyrics(self, text: str) -> tuple[list, list]:
        """从歌词内容中提取实体和三元组，保留原始内容特征"""
        import re

        # 解析原始内容标记
        original_pattern = r'\[ORIGINAL:([^\]]+)\](.*?)\[/ORIGINAL\]'
        entities = []
        triples = []

        # 分离原始内容和描述性内容
        original_blocks = []
        clean_text_parts = []

        # 提取所有原始内容块
        last_end = 0
        for match in re.finditer(original_pattern, text, re.DOTALL):
            # 添加标记前的文本
            clean_text_parts.append(text[last_end:match.start()])

            label = match.group(1)
            content = match.group(2).strip()
            original_blocks.append((label, content))

            last_end = match.end()

        # 添加最后一部分文本
        clean_text_parts.append(text[last_end:])

        # 合并非原始内容的文本
        clean_text = ''.join(clean_text_parts).strip()

        # 对描述性内容进行标准实体提取
        if clean_text:
            try:
                from src.llm_models.utils_model import LLMRequest
                from src.chat.knowledge.ie_process import info_extract_from_str
                import model_config

                ner_llm = LLMRequest(
                    model_set=model_config.model_task_config.lpmm_entity_extract,
                    request_type="memories.lpmm.ner",
                )
                rdf_llm = LLMRequest(
                    model_set=model_config.model_task_config.lpmm_rdf_build,
                    request_type="memories.lpmm.rdf",
                )

                desc_entities, desc_triples = info_extract_from_str(ner_llm, rdf_llm, clean_text)
                if desc_entities:
                    entities.extend(desc_entities)
                if desc_triples:
                    triples.extend(desc_triples)
            except Exception as e:
                logger.warning("描述性内容实体提取失败: %s", e)

        # 为原始内容块创建结构化实体和三元组
        for label, content in original_blocks:
            # 创建表示原始内容的实体
            content_entity = f"原始内容_{label}_{hash(content) % 10000}"
            entities.append(content_entity)

            # 创建描述性三元组
            triples.append({
                "subject": content_entity,
                "predicate": "属于类型",
                "object": label
            })

            triples.append({
                "subject": content_entity,
                "predicate": "包含内容",
                "object": content[:100] + "..." if len(content) > 100 else content
            })

        return entities, triples

    def _build_personality_prompt(self) -> str:
        """构建主人设提示，用于观点总结。"""
        bot_name = global_config.bot.nickname
        if global_config.bot.alias_names:
            bot_nickname = f",也有人叫你{','.join(global_config.bot.alias_names)}"
        else:
            bot_nickname = ""

        # 获取基础personality
        prompt_personality = global_config.personality.personality

        # 检查是否需要随机替换为状态
        if (
            global_config.personality.states
            and global_config.personality.state_probability > 0
            and random.random() < global_config.personality.state_probability
        ):
            # 随机选择一个状态替换personality
            selected_state = random.choice(global_config.personality.states)
            prompt_personality = selected_state

        prompt_personality = f"{prompt_personality};"

        # 根据配置决定是否包含回复风格
        include_reply_style = self.get_config("viewpoint.include_reply_style", True)
        if include_reply_style and global_config.personality.reply_style:
            reply_style = global_config.personality.reply_style
            return f"你的名字是{bot_name}{bot_nickname}，你{prompt_personality}你的回复风格是：{reply_style}"
        else:
            return f"你的名字是{bot_name}{bot_nickname}，你{prompt_personality}"

    async def _generate_viewpoint_summary(self, original_summary: str, title: str) -> str:
        """基于主人设生成观点总结。"""
        if not original_summary.strip():
            return ""

        # 检查是否启用观点总结
        if not self.get_config("viewpoint.enabled", False):
            return ""

        personality_prompt = self._build_personality_prompt()
        max_chars = self.get_config("viewpoint.max_chars", 500)

        prompt = (
            f"{personality_prompt}\n\n"
            "请基于以上的人设，对下面的聊天记录摘要进行观点总结。\n"
            "要求：\n"
            "1. 以第一人称表达我的观点和感受\n"
            "2. 重点关注这段对话中的重要内容和我个人的反应\n"
            "3. 格式清晰，使用中小段落，避免过长的句子\n"
            "4. 保持温暖、真实的语气\n"
            "5. 控制在 {max_chars} 字符以内\n\n"
            f"聊天记录摘要：\n{original_summary}\n\n"
            "观点总结："
        )

        task_config = self._get_task_config()
        temperature = self.get_config("summary.temperature", 0.7)  # 观点总结使用稍高的温度
        ok, response, _, model_name = await llm_api.generate_with_model(
            prompt,
            task_config,
            request_type=f"memories.viewpoint.{title}",
            temperature=temperature,
        )
        if not ok:
            logger.error("%s 的观点总结 LLM 调用失败。", title)
            return ""
        viewpoint = response.strip()
        if not viewpoint:
            logger.warning("%s 的观点总结为空。", title)
            return ""

        viewpoint = self._truncate_text(viewpoint, max_chars)
        logger.info("已生成观点总结 (%s)，使用模型 %s。", title, model_name)
        return viewpoint

    async def _summarize_text(self, content: str, max_chars: int, title: str) -> str:
        if not content.strip():
            return ""

        # 预处理内容，过滤掉插件回复和系统消息
        content = self._preprocess_chat_content(content)
        if not content.strip():
            return ""

        task_config = self._get_task_config()
        prompt = self._build_prompt(title, max_chars, content)
        temperature = self.get_config("summary.temperature", None)
        ok, response, _, model_name = await llm_api.generate_with_model(
            prompt,
            task_config,
            request_type=f"memories.summary.{title}",
            temperature=temperature,
        )
        if not ok:
            logger.error("%s 的 LLM 摘要失败。", title)
            return ""
        summary = response.strip()
        if not summary:
            logger.warning("%s 的摘要为空。", title)
            return ""
        
        # 预处理分段
        summary = self._preprocess_summary_for_lpmm(summary)
        summary = self._truncate_text(summary, max_chars)

        # 生成观点总结（如果启用）
        viewpoint_summary = await self._generate_viewpoint_summary(summary, title)
        if viewpoint_summary:
            # 将观点总结附加到原始摘要后
            summary = f"{summary}\n\n观点总结：\n{viewpoint_summary}"

        logger.info("已生成摘要 (%s)，使用模型 %s。", title, model_name)
        return summary

    async def _build_entries_text(self, entries: list[dict], tz) -> str:
        lines = []
        for entry in entries:
            raw_file = entry.get("raw_file")
            if not raw_file:
                continue
            path = self._resolve_entry_path(raw_file)
            if not path.exists():
                continue
            try:
                text = await asyncio.to_thread(path.read_text, "utf-8")
            except Exception:
                continue
            text = text.strip()
            created_at = float(entry.get("created_at", 0) or 0)
            if tz:
                date_label = datetime.fromtimestamp(created_at, tz).strftime("%Y-%m-%d")
            else:
                date_label = datetime.fromtimestamp(created_at).strftime("%Y-%m-%d")
            lines.append(f"日期 {date_label}:\n{text}")
        # 使用双换行连接，确保 LPMM 将不同日期的摘要视为独立段落
        return "\n\n".join(lines)

    async def _persist_summary(
        self,
        chat_id: str,
        kind: str,
        level: int,
        summary: str,
        start_ts: float,
        end_ts: float,
        max_chars: int,
        tz,
        import_flag: bool = True,
    ) -> dict | None:
        self._ensure_dirs()
        # 按需导入 hash 函数，避免在模块导入时加载大型依赖
        try:
            from src.chat.knowledge.utils.hash import get_sha256
        except Exception:
            import hashlib
            def get_sha256(text: str) -> str:
                return hashlib.sha256(text.encode("utf-8")).hexdigest()
        created_at = _now_ts()
        safe_id = _sanitize_filename(chat_id)
        stamp = int(created_at)
        raw_path = self._raw_dir / f"{kind}_{safe_id}_{stamp}.txt"
        openie_path = self._openie_dir / f"{kind}_{safe_id}_{stamp}_openie.json"
        if tz:
            start_label = datetime.fromtimestamp(start_ts, tz).strftime("%Y-%m-%d %H:%M")
            end_label = datetime.fromtimestamp(end_ts, tz).strftime("%Y-%m-%d %H:%M")
        else:
            start_label = datetime.fromtimestamp(start_ts).strftime("%Y-%m-%d %H:%M")
            end_label = datetime.fromtimestamp(end_ts).strftime("%Y-%m-%d %H:%M")
        
        # 构建元数据头部，并与正文用空行分隔
        meta = f"聊天会话 {chat_id} ({start_label} 至 {end_label})"
        full_text = f"{meta}\n\n{summary}"
        full_text = self._truncate_text(full_text, max_chars)
        
        # 写入原始文件，添加错误处理
        try:
            await asyncio.to_thread(raw_path.write_text, full_text.strip() + "\n", "utf-8")
        except Exception as exc:
            logger.error("写入原始文件失败: %s, 路径: %s", exc, raw_path, exc_info=True)
            return None
        
        # 检查是否需要确认导入
        confirm_import = bool(self.get_config("summary.confirm_import", False))
        if import_flag and confirm_import:
            # 需要确认导入，创建待确认任务
            task_id = f"{kind}_{safe_id}_{stamp}"
            task = PendingConfirmTask(
                task_id=task_id,
                task_type=kind,
                chat_id=chat_id,
                summary_text=summary,
                raw_path=raw_path,
                openie_path=openie_path,
                created_at=created_at,
            )
            # 设置确认超时时间
            confirm_timeout = int(self.get_config("summary.confirm_timeout", 300))
            task.confirm_timeout = created_at + confirm_timeout

            await self._add_pending_confirm_task(task)
            await self._send_confirm_message(task)
            logger.info("摘要已保存，等待管理员确认导入: %s", task_id)
        elif import_flag:
            # 直接导入
            ok = await self._import_raw_summary(raw_path, openie_path)
            if not ok:
                # 清理已写入的文件，避免残留
                try:
                    def _cleanup():
                        if raw_path.exists():
                            raw_path.unlink()
                    await asyncio.to_thread(_cleanup)
                    logger.debug("已清理失败的原始文件: %s", raw_path)
                except Exception as cleanup_exc:
                    logger.warning("清理失败文件时出错: %s", cleanup_exc)
                return None
        else:
            logger.info("跳过知识库导入，仅保存摘要文件: %s", raw_path)
        summary_hash = get_sha256(full_text)
        entry_id = f"{kind}-{safe_id}-{stamp}-{summary_hash[:8]}"
        return {
            "id": entry_id,
            "chat_id": chat_id,
            "kind": kind,
            "level": level,
            "hash": summary_hash,
            "raw_file": self._relpath(raw_path),
            "openie_file": self._relpath(openie_path),
            "created_at": created_at,
            "source_start": start_ts,
            "source_end": end_ts,
            "deleted": False,
        }

    async def _run_script(self, script_path: Path, script_args: list[str], label: str) -> bool:
        if not script_path.exists():
            logger.error("缺少脚本: %s", script_path)
            return False
        python_exe = sys.executable or "python"
        proc = await asyncio.create_subprocess_exec(
            python_exe,
            str(script_path),
            *script_args,
            cwd=str(self._root_dir),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            logger.error("%s 失败 (代码 %s)。", label, proc.returncode)
            if stdout:
                logger.error("%s 标准输出: %s", label, stdout.decode("utf-8", errors="ignore")[:2000])
            if stderr:
                logger.error("%s 标准错误: %s", label, stderr.decode("utf-8", errors="ignore")[:2000])
            return False
        if stdout:
            logger.info("%s 标准输出: %s", label, stdout.decode("utf-8", errors="ignore")[:1000])
        if stderr:
            logger.warning("%s 标准错误: %s", label, stderr.decode("utf-8", errors="ignore")[:1000])
        return True

    async def _import_raw_summary(self, raw_path: Path, openie_path: Path) -> bool:
        """内部实现信息提取与导入，不再依赖外部脚本的命令行参数。
        为避免在事件循环中阻塞，将耗时的同步工作封装到线程中执行。
        """
        # 限制并发，避免线程池被大量耗尽
        acquired = False
        try:
            await self._heavy_semaphore.acquire()
            acquired = True
            return await asyncio.to_thread(self._import_raw_summary_blocking, raw_path, openie_path)
        except Exception as e:
            logger.error("导入摘要到 LPMM 时发生异常: %s", e, exc_info=True)
            return False
        finally:
            if acquired:
                try:
                    self._heavy_semaphore.release()
                except Exception:
                    pass

    def _release_heavy_modules(self, module_prefixes: list[str]) -> None:
        """尝试卸载按需导入的模块并触发 GC，以便释放 Python 层引用和部分可回收的内存。
        注意：无法保证释放底层 C 扩展分配的内存；对共享模块请谨慎执行卸载。
        """
        try:
            import sys
            import gc
            # 删除 sys.modules 中以这些前缀开头的模块
            to_delete = [name for name in list(sys.modules.keys()) if any(name.startswith(p) for p in module_prefixes)]
            for name in to_delete:
                try:
                    del sys.modules[name]
                except Exception:
                    pass
            gc.collect()
        except Exception:
            # 不应抛出异常到业务流程
            logger.debug("卸载重型模块时发生异常", exc_info=True)

    def _import_raw_summary_blocking(self, raw_path: Path, openie_path: Path) -> bool:
        """同步、阻塞式的导入实现，供后台线程执行。"""
        # 按需导入可能占用大量内存的模块（LLM、Embedding、KG、IE）
        module_prefixes = [
            "src.llm_models.utils_model",
            "src.chat.knowledge.ie_process",
            "src.chat.knowledge.open_ie",
            "src.chat.knowledge.embedding_store",
            "src.chat.knowledge.kg_manager",
            "src.chat.knowledge.utils.hash",
        ]
        try:
            from src.config.config import global_config, model_config
            from src.llm_models.utils_model import LLMRequest
            from src.chat.knowledge.ie_process import info_extract_from_str
            from src.chat.knowledge.open_ie import OpenIE
            from src.chat.knowledge.embedding_store import EmbeddingManager
            from src.chat.knowledge.kg_manager import KGManager
            from src.chat.knowledge.utils.hash import get_sha256
            # 记录已按需导入的模块前缀，任务完成后可尝试卸载
            try:
                self._loaded_heavy_module_prefixes.update(module_prefixes)
            except Exception:
                # 忽略在非实例上下文或其他意外情况
                pass
        except Exception as exc:
            logger.error("导入 LPMM 相关模块失败: %s", exc, exc_info=True)
            return False
        # 1. 读取原始文本
        try:
            text = raw_path.read_text(encoding="utf-8").strip()
        except Exception as e:
            logger.error("读取原始文件失败: %s, 路径: %s", e, raw_path, exc_info=True)
            return False
        if not text:
            return False

        # 2. 初始化 LLM 请求
        ner_llm = LLMRequest(
            model_set=model_config.model_task_config.lpmm_entity_extract,
            request_type="memories.lpmm.ner",
        )
        rdf_llm = LLMRequest(
            model_set=model_config.model_task_config.lpmm_rdf_build,
            request_type="memories.lpmm.rdf",
        )

        # 3. 预处理文本：识别原始内容并进行特殊处理
        has_original_content = '[ORIGINAL:' in text

        if has_original_content:
            # 对于包含原始内容的文本，使用增强的实体提取逻辑
            entities, triples = self._extract_entities_from_lyrics(text)
            if not entities or not triples:
                # 如果增强提取失败，回退到标准提取
                entities, triples = info_extract_from_str(ner_llm, rdf_llm, text)
        else:
            # 标准信息提取
            entities, triples = info_extract_from_str(ner_llm, rdf_llm, text)
        if entities is None or triples is None:
            logger.error("LPMM 信息提取失败。entities=%s, triples=%s", entities, triples)
            return False

        logger.debug("信息提取结果: entities类型=%s, 长度=%s, triples类型=%s, 长度=%s",
                    type(entities), len(entities) if hasattr(entities, '__len__') else 'N/A',
                    type(triples), len(triples) if hasattr(triples, '__len__') else 'N/A')

        # 4. 构建 OpenIE 对象并保存中间文件
        pg_hash = get_sha256(text)
        doc_item = {
            "idx": pg_hash,
            "passage": text,
            "extracted_entities": entities,
            "extracted_triples": triples,
        }

        sum_chars = sum(len(e) for e in entities)
        sum_words = sum(len(e.split()) for e in entities)
        num_ent = len(entities)
        avg_chars = round(sum_chars / num_ent, 4) if num_ent else 0
        avg_words = round(sum_words / num_ent, 4) if num_ent else 0

        openie_obj = OpenIE([doc_item], avg_chars, avg_words)
        try:
            openie_path.write_text(
                json.dumps(openie_obj.__dict__, ensure_ascii=False, indent=4),
                encoding="utf-8"
            )
        except Exception as exc:
            logger.error("写入 OpenIE 文件失败: %s, 路径: %s", exc, openie_path, exc_info=True)
            return False

        # 5. 导入到向量库与知识图谱
        embed_manager = EmbeddingManager()
        try:
            embed_manager.load_from_file()
        except Exception as exc:
            logger.warning("加载 EmbeddingManager 失败: %s", exc, exc_info=True)
            # 继续执行，可能是首次运行

        kg_manager = KGManager()
        try:
            kg_manager.load_from_file()
        except Exception as exc:
            logger.warning("加载 KGManager 失败: %s", exc, exc_info=True)
            # 继续执行，可能是首次运行

        paragraph_key = f"paragraph-{pg_hash}"
        if (paragraph_key in embed_manager.stored_pg_hashes and
            pg_hash in kg_manager.stored_paragraph_hashes):
            logger.info("段落已存在于知识库中，跳过导入。")
            # 尝试清理临时持有的引用并卸载模块
            try:
                if hasattr(embed_manager, "close"):
                    try:
                        embed_manager.close()
                    except Exception:
                        pass
                if hasattr(kg_manager, "close"):
                    try:
                        kg_manager.close()
                    except Exception:
                        pass
                # 删除本地引用以便 GC 回收
                del embed_manager
                del kg_manager
                del openie_obj
                # 尝试卸载相关模块
                try:
                    self._release_heavy_modules(module_prefixes)
                except Exception:
                    pass
            except Exception:
                pass
            return True

        raw_paragraphs = {pg_hash: text}
        triple_list_data = {pg_hash: triples}

        # 尝试在独立子进程中执行索引重建与 KG 构建，以确保重型内存分配在子进程中完成，
        # 任务完成后子进程退出可将内存返还给操作系统。
        processed_ok = False
        try:
            self._log_memory_state("before_subprocess_attempt", note=f"pg_hash={pg_hash}")
        except Exception:
            pass

        # 直接使用subprocess执行，避免Pickling问题
        try:
            logger.debug("准备使用子进程脚本执行任务")

            # 写临时 JSON 输入文件
            import tempfile

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as rawf:
                rawf.write(json.dumps(raw_paragraphs, ensure_ascii=False))
                raw_path_tmp = rawf.name
            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as triplef:
                triplef.write(json.dumps(triple_list_data, ensure_ascii=False))
                triple_path_tmp = triplef.name

            worker_script = Path(self.plugin_dir) / "subprocess_worker.py"
            import subprocess

            try:
                proc = subprocess.run(
                    [sys.executable or "python", str(worker_script), str(raw_path_tmp), str(triple_path_tmp)],
                    cwd=str(self._root_dir),
                    capture_output=True,
                    text=True,
                    encoding='utf-8',  # 指定UTF-8编码以避免Windows GBK解码错误
                    timeout=300,  # 5分钟超时
                )
                if proc.returncode != 0:
                    logger.warning(
                        "子进程脚本返回非零代码 %s; stdout=%s stderr=%s",
                        proc.returncode,
                        (proc.stdout or "")[:2000],
                        (proc.stderr or "")[:2000],
                    )
                    processed_ok = False
                else:
                    logger.info("子进程脚本执行成功，stdout=%s", (proc.stdout or "")[:1000])
                    processed_ok = True
            except subprocess.TimeoutExpired:
                logger.warning("子进程重建/构建超时")
                try:
                    self._log_memory_state("subprocess_timeout", note=f"pg_hash={pg_hash}")
                except Exception:
                    pass
                processed_ok = False
            except Exception as run_exc:
                logger.warning("子进程脚本执行失败: %s", run_exc, exc_info=True)
                processed_ok = False
        except Exception:
            # 如果不能导入或使用进程池，则回退到线程内执行
            processed_ok = False

        # 如果子进程执行失败或不可用，则回退到线程内执行（兼容性保障）
        if not processed_ok:
            try:
                embed_manager = EmbeddingManager()
                try:
                    embed_manager.load_from_file()
                except Exception as exc:
                    logger.warning("加载 EmbeddingManager 失败: %s", exc, exc_info=True)
                kg_manager = KGManager()
                try:
                    kg_manager.load_from_file()
                except Exception as exc:
                    logger.warning("加载 KGManager 失败: %s", exc, exc_info=True)

                embed_manager.store_new_data_set(raw_paragraphs, triple_list_data)
                embed_manager.rebuild_faiss_index()
                embed_manager.save_to_file()

                kg_manager.build_kg(triple_list_data, embed_manager)
                kg_manager.save_to_file()
            except Exception as exc:
                logger.error("回退到线程执行索引/KG 构建失败: %s", exc, exc_info=True)
                return False

        # 使节点计数缓存失效，下次获取时会重新计算
        try:
            self._invalidate_node_count_cache()
        except Exception:
            # 在后台线程中调用实例方法，保护性捕获异常
            logger.debug("后台线程调用 _invalidate_node_count_cache 失败", exc_info=True)

        logger.info("成功将摘要导入 LPMM 知识库。")
        # 尝试清理持有的大对象并卸载按需导入的模块
        try:
            if 'embed_manager' in locals():
                try:
                    if hasattr(embed_manager, "close"):
                        embed_manager.close()
                except Exception:
                    pass
            if 'kg_manager' in locals():
                try:
                    if hasattr(kg_manager, "close"):
                        kg_manager.close()
                except Exception:
                    pass
            # 删除本地引用
            for name in ("embed_manager", "kg_manager", "openie_obj", "entities", "triples", "doc_item"):
                if name in locals():
                    try:
                        del locals()[name]
                    except Exception:
                        pass
            try:
                self._release_heavy_modules(module_prefixes)
            except Exception:
                pass
        except Exception:
            pass
        return True

    async def _run_due(self) -> None:
        if not self._is_enabled():
            logger.debug("_run_due: 插件未启用，跳过")
            return
        if not self._lpmm_enabled():
            logger.info("LPMM 已禁用；跳过摘要。")
            return
        tz = self._get_timezone()
        now = datetime.now(tz)
        streams = self._select_streams()
        logger.info("_run_due: 调度器运行中，当前时间: %s, 找到 %d 个流", now.strftime("%H:%M:%S"), len(streams))
        if not streams:
            logger.debug("_run_due: 没有找到符合条件的流")
            return
        daily_anchor = self._daily_anchor(now)
        logger.info("_run_due: 执行每日摘要，锚点时间: %s", daily_anchor.strftime("%H:%M:%S"))
        await self.run_daily(streams=streams, manual=False, anchor=daily_anchor)
        if bool(self.get_config("schedule.enable_weekly", True)):
            weekly_anchor = self._weekly_anchor(now)
            logger.info("_run_due: 执行每周摘要，锚点时间: %s", weekly_anchor.strftime("%H:%M:%S"))
            await self.run_weekly(streams=streams, manual=False, anchor=weekly_anchor)

    async def run_forever(self, trigger_message: MaiMessages, import_flag: bool = True) -> None:
        """手动触发的"永远的记忆"逻辑。"""
        chat_id = None
        try:
            if not self._is_enabled() or not self._lpmm_enabled():
                logger.debug("插件未启用或 LPMM 未启用，跳过永远的记忆处理")
                return

            chat_id = trigger_message.chat_stream.stream_id if trigger_message.chat_stream else None
            if not chat_id:
                logger.warning("永远的记忆触发消息缺少 chat_id")
                return

            logger.info("开始处理永远的记忆请求，chat_id: %s", chat_id)

            async with self._run_lock:
                await self._precheck_performance(manual=True)
                lookback = int(self.get_config("forever.lookback_messages", 20))
                forever_max_chars = int(self.get_config("summary.forever_max_chars", 1000))
                
                # 获取最近的消息（按条数回溯）
                end_ts = _now_ts()
                fetch_limit = max(lookback * 3, lookback)
                messages = message_api.get_messages_before_time_in_chat(
                    chat_id,
                    end_ts,
                    limit=fetch_limit,
                    filter_mai=False,  # 永远的记忆通常包含机器人的回复
                )
                filter_command = bool(self.get_config("summary.filter_command", True))
                if filter_command:
                    messages = [msg for msg in messages if not getattr(msg, "is_command", False)]
                
                if not messages:
                    logger.info("未找到符合条件的消息，chat_id: %s", chat_id)
                    return
                
                # 如果消息太多，只取最后 lookback 条
                messages = sorted(messages, key=lambda msg: msg.time)
                if len(messages) > lookback:
                    messages = messages[-lookback:]
                start_ts = float(messages[0].time or end_ts)
                end_ts = float(messages[-1].time or end_ts)

                logger.debug("处理 %d 条消息，时间范围: %s - %s", len(messages), start_ts, end_ts)

                readable = message_api.build_readable_messages_to_str(
                    messages,
                    replace_bot_name=True,
                    timestamp_mode="relative",
                    truncate=True,
                )
                
                summary = await self._summarize_text(readable, forever_max_chars, "forever_memory")
                if not summary:
                    logger.warning("永远的记忆摘要生成失败，chat_id: %s", chat_id)
                    return
                
                entry = await self._persist_summary(
                    chat_id=chat_id,
                    kind="forever",
                    level=3, # 级别 3 代表永远的记忆
                    summary=summary,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    max_chars=forever_max_chars,
                    tz=self._get_timezone(),
                    import_flag=import_flag,
                )
                
                if entry:
                    state = await self._load_state()
                    state["entries"].append(entry)
                    await self._save_state(state)
                    try:
                        from src.chat.knowledge import lpmm_start_up
                    except Exception:
                        lpmm_start_up = None
                    if lpmm_start_up:
                        lpmm_start_up()
                    logger.info("已成功登记永远的记忆: %s, chat_id: %s", entry["id"], chat_id)
                    # 可以选择发送一个确认消息
                    try:
                        await trigger_message.answer("好的，这段重要的对话我已经存入永远的记忆中了。")
                    except Exception as e:
                        logger.warning("发送确认消息失败: %s", e)
                else:
                    logger.warning("永远的记忆持久化失败，chat_id: %s", chat_id)
                
                # 运行结束后检查性能
                is_dangerous, desc = self._check_performance()
                if is_dangerous:
                    await self._alert_admin(desc)
                    
        except asyncio.CancelledError:
            logger.info("永远的记忆任务被取消，chat_id: %s", chat_id)
            raise
        except Exception as exc:
            logger.error("永远的记忆处理失败，chat_id: %s, 错误: %s", chat_id, exc, exc_info=True)
            # 尝试发送错误通知
            try:
                if trigger_message and chat_id:
                    await trigger_message.answer("抱歉，处理记忆时发生了错误，请稍后重试。")
            except Exception as e:
                logger.warning("发送错误通知失败: %s", e)

    async def run_daily(self, streams=None, manual: bool = False, anchor: datetime | None = None, import_flag: bool = True) -> None:
        if not self._is_enabled():
            return
        if not self._lpmm_enabled():
            logger.info("LPMM 已禁用；跳过每日摘要。")
            return
        
        if self._needs_approval_event.is_set() and not manual:
            await self._alert_admin("由于之前的性能预警，每日摘要任务已暂停，等待管理员确认。")
            return

        if not await self._precheck_performance(manual=manual):
            return

        async with self._run_lock:
            tz = self._get_timezone()
            now = datetime.now(tz)
            anchor_dt = anchor or self._daily_anchor(now)
            anchor_ts = anchor_dt.timestamp()
            end_ts = _now_ts() if manual else anchor_ts

            target_streams = streams or self._select_streams()
            if not target_streams:
                return

            state = await self._load_state()
            changed = False
            refresh_needed = False

            min_messages = int(self.get_config("summary.min_messages", 8))
            max_input_chars = int(self.get_config("summary.max_input_chars", 8000))
            daily_max_chars = int(self.get_config("summary.daily_max_chars", 800))
            filter_bot = bool(self.get_config("summary.filter_bot", False))
            filter_command = bool(self.get_config("summary.filter_command", True))
            truncate_messages = bool(self.get_config("summary.truncate_messages", True))
            auto_import = import_flag if manual else bool(self.get_config("summary.auto_import", True))

            for stream in target_streams:
                chat_id = getattr(stream, "stream_id", None)
                if not chat_id:
                    continue
                if not self._stream_allowed(stream):
                    continue
                last_run = float(state["last_daily_run"].get(chat_id, 0) or 0)
                if not manual and last_run >= anchor_ts:
                    continue
                if manual:
                    start_ts = last_run if last_run > 0 else end_ts - DAY_SECONDS
                else:
                    # 对于首次运行，使用 end_ts - DAY_SECONDS 确保获取最近24小时的消息
                    # 对于非首次运行，使用 last_run 确保不遗漏消息
                    if last_run > 0:
                        start_ts = last_run
                    else:
                        start_ts = end_ts - DAY_SECONDS
                if start_ts >= end_ts:
                    state["last_daily_run"][chat_id] = end_ts
                    changed = True
                    continue
                messages = message_api.get_messages_by_time_in_chat(
                    chat_id,
                    start_ts,
                    end_ts,
                    filter_mai=filter_bot,
                    filter_command=filter_command,
                )
                if len(messages) < min_messages:
                    state["last_daily_run"][chat_id] = end_ts
                    changed = True
                    continue
                readable = message_api.build_readable_messages_to_str(
                    messages,
                    replace_bot_name=True,
                    timestamp_mode="relative",
                    truncate=truncate_messages,
                )
                readable = self._truncate_text(readable, max_input_chars)
                summary = await self._summarize_text(readable, daily_max_chars, "daily")
                if not summary:
                    continue
                entry = await self._persist_summary(
                    chat_id=chat_id,
                    kind="daily",
                    level=0,
                    summary=summary,
                    start_ts=start_ts,
                    end_ts=end_ts,
                    max_chars=daily_max_chars,
                    tz=tz,
                    import_flag=auto_import,
                )
                if not entry:
                    continue
                state["entries"].append(entry)
                state["last_daily_run"][chat_id] = end_ts
                changed = True
                refresh_needed = True

            capacity_changed, capacity_refresh = await self._enforce_capacity(state)
            refresh_needed = refresh_needed or capacity_refresh

            if refresh_needed:
                try:
                    from src.chat.knowledge import lpmm_start_up
                except Exception:
                    lpmm_start_up = None
                if lpmm_start_up:
                    lpmm_start_up()
            if changed or capacity_changed:
                await self._save_state(state)
            
            # 运行结束后检查性能
            is_dangerous, desc = self._check_performance()
            if is_dangerous:
                await self._alert_admin(desc)

    async def run_weekly(self, streams=None, manual: bool = False, anchor: datetime | None = None, import_flag: bool = True) -> None:
        if not self._is_enabled():
            return
        if not manual and not bool(self.get_config("schedule.enable_weekly", True)):
            return
        if not self._lpmm_enabled():
            logger.info("LPMM 已禁用；跳过每周摘要。")
            return
        
        if self._needs_approval_event.is_set() and not manual:
            await self._alert_admin("由于之前的性能预警，每周摘要任务已暂停，等待管理员确认。")
            return

        if not await self._precheck_performance(manual=manual):
            return

        async with self._run_lock:
            tz = self._get_timezone()
            now = datetime.now(tz)
            anchor_dt = anchor or self._weekly_anchor(now)
            anchor_ts = anchor_dt.timestamp()
            end_ts = _now_ts() if manual else anchor_ts

            target_streams = streams or self._select_streams()
            if not target_streams:
                return

            state = await self._load_state()
            changed = False
            refresh_needed = False

            max_input_chars = int(self.get_config("summary.max_input_chars", 8000))
            weekly_max_chars = int(self.get_config("summary.weekly_max_chars", 1200))
            auto_import = import_flag if manual else bool(self.get_config("summary.auto_import", True))

            for stream in target_streams:
                chat_id = getattr(stream, "stream_id", None)
                if not chat_id:
                    continue
                if not self._stream_allowed(stream):
                    continue
                last_run = float(state["last_weekly_run"].get(chat_id, 0) or 0)
                if not manual and last_run >= anchor_ts:
                    continue
                # 限制时间窗口为一周，确保只处理最近一周的每日摘要
                # 如果 last_run 很早（比如一个月前），限制在一周内
                # 如果 last_run 在一周内，使用 last_run 确保不遗漏消息
                week_start = end_ts - WEEK_SECONDS
                if last_run > 0 and last_run >= week_start:
                    since_ts = last_run  # 不遗漏，且在一周内
                else:
                    since_ts = week_start  # 首次运行或 last_run 太早，使用一周前
                daily_entries = [
                    e
                    for e in state["entries"]
                    if e.get("chat_id") == chat_id
                    and e.get("level") == 0
                    and not e.get("deleted")
                    and float(e.get("created_at", 0) or 0) >= since_ts
                ]
                daily_entries = sorted(daily_entries, key=lambda e: e.get("created_at", 0))
                if len(daily_entries) < MIN_WEEKLY_DAILY_ENTRIES:
                    state["last_weekly_run"][chat_id] = end_ts
                    changed = True
                    continue
                daily_entries = daily_entries[-MIN_WEEKLY_DAILY_ENTRIES:]
                entries_text = await self._build_entries_text(daily_entries, tz)
                if not entries_text:
                    state["last_weekly_run"][chat_id] = end_ts
                    changed = True
                    continue
                entries_text = self._truncate_text(entries_text, max_input_chars)
                summary = await self._summarize_text(entries_text, weekly_max_chars, "weekly")
                if not summary:
                    continue
                entry = await self._persist_summary(
                    chat_id=chat_id,
                    kind="weekly",
                    level=1,
                    summary=summary,
                    start_ts=since_ts,
                    end_ts=end_ts,
                    max_chars=weekly_max_chars,
                    tz=tz,
                    import_flag=auto_import,
                )
                if not entry:
                    continue
                state["entries"].append(entry)
                state["last_weekly_run"][chat_id] = end_ts
                changed = True
                refresh_needed = True
                if await self._delete_entries(daily_entries):
                    refresh_needed = True
                    changed = True

            capacity_changed, capacity_refresh = await self._enforce_capacity(state)
            refresh_needed = refresh_needed or capacity_refresh

            if refresh_needed:
                try:
                    from src.chat.knowledge import lpmm_start_up
                except Exception:
                    lpmm_start_up = None
                if lpmm_start_up:
                    lpmm_start_up()
            if changed or capacity_changed:
                await self._save_state(state)
            
            # 运行结束后检查性能
            is_dangerous, desc = self._check_performance()
            if is_dangerous:
                await self._alert_admin(desc)

    async def _get_node_count(self, force_refresh: bool = False) -> int:
        """获取 KG 节点数，使用缓存避免重复加载。此方法为异步，内部在后台线程执行阻塞 I/O。"""
        if not self._lpmm_enabled():
            return 0
        
        # 检查缓存是否有效
        now = _now_ts()
        if (not force_refresh and 
            self._cached_node_count is not None and 
            now - self._node_count_cache_time < self._node_count_cache_ttl):
            return self._cached_node_count

        # 刷新缓存（在后台线程中执行阻塞操作）
        try:
            def _load_count():
                from src.chat.knowledge.kg_manager import KGManager
                kg_manager = KGManager()
                kg_manager.load_from_file()
                return len(kg_manager.graph.get_node_list())

            count = await asyncio.to_thread(_load_count)
            self._cached_node_count = count
            self._node_count_cache_time = now
            logger.debug("KG 节点数已更新: %d", count)
            return count
        except Exception as exc:
            logger.warning("加载 KG 获取节点数失败: %s", exc, exc_info=True)
            # 如果缓存存在，返回缓存值；否则返回 0
            return self._cached_node_count if self._cached_node_count is not None else 0
    
    def _invalidate_node_count_cache(self) -> None:
        """使节点计数缓存失效。"""
        self._cached_node_count = None
        self._node_count_cache_time = 0.0

    async def _delete_entries(self, entries: list[dict]) -> bool:
        hashes = [e.get("hash") for e in entries if e.get("hash")]
        if not hashes:
            return False
        self._ensure_dirs()
        stamp = int(_now_ts())
        delete_path = self._delete_dir / f"delete_{stamp}.txt"
        try:
            delete_path.write_text("\n".join(hashes) + "\n", encoding="utf-8")
        except Exception as exc:
            logger.error("写入删除文件失败: %s, 路径: %s", exc, delete_path, exc_info=True)
            return False
        delete_script = self._root_dir / "scripts" / "delete_lpmm_items.py"
        ok = await self._run_script(
            delete_script,
            [
                "--hash-file",
                str(delete_path),
                "--delete-entities",
                "--delete-relations",
                "--remove-orphan-entities",
                "--yes",
                "--non-interactive",
            ],
            "删除 LPMM 条目",
        )
        if ok:
            for entry in entries:
                entry["deleted"] = True
                entry["deleted_at"] = _now_ts()
        return ok

    async def _enforce_capacity(self, state: dict) -> tuple[bool, bool]:
        max_paragraphs = int(self.get_config("capacity.max_paragraphs", 0))
        max_nodes = int(self.get_config("capacity.max_nodes", 0))
        enable_level2 = bool(self.get_config("capacity.enable_level2", True))

        # 过滤掉已删除的条目，且只针对 level < 3 的条目进行容量管理
        # level 3 是“永远的记忆”，不参与自动清理
        active_entries = [e for e in state["entries"] if not e.get("deleted")]
        cleanup_candidates = [e for e in active_entries if e.get("level", 0) < 3]
        
        if not cleanup_candidates:
            return False, False

        nodes_exceeded = False
        node_count = 0
        if max_nodes > 0:
            node_count = await self._get_node_count()
            nodes_exceeded = node_count > max_nodes

        # 注意：这里判断是否超限时，通常是看总数，但删除时只删 candidates
        count_exceeded = max_paragraphs > 0 and len(active_entries) > max_paragraphs
        if not count_exceeded and not nodes_exceeded:
            return False, False

        delete_target = 0
        if count_exceeded:
            delete_target = len(active_entries) - max_paragraphs
        elif nodes_exceeded:
            delete_target = min(MAX_DELETE_BATCH_SIZE, len(cleanup_candidates))

        changed = False
        refresh_needed = False

        if delete_target > 0:
            # 优先删除每日摘要 (Level 0)
            daily_entries = sorted(
                [e for e in cleanup_candidates if e.get("level") == 0],
                key=lambda e: e.get("created_at", 0),
            )
            if daily_entries:
                to_delete = daily_entries[:delete_target]
                if await self._delete_entries(to_delete):
                    changed = True
                    refresh_needed = True
            
            # 重新计算
            active_entries = [e for e in state["entries"] if not e.get("deleted")]
            cleanup_candidates = [e for e in active_entries if e.get("level", 0) < 3]
            
            # 如果是因为节点数超限而删除，重新检查节点数
            if nodes_exceeded and max_nodes > 0:
                node_count = await self._get_node_count()
                nodes_exceeded = node_count > max_nodes

        still_exceeded = False
        if max_paragraphs > 0 and len(active_entries) > max_paragraphs:
            still_exceeded = True
        if max_nodes > 0 and nodes_exceeded:
            still_exceeded = True

        if still_exceeded and enable_level2:
            # 尝试对 Level 1 进行压缩
            level2_changed, level2_refresh = await self._maybe_build_level2(state, cleanup_candidates)
            if level2_changed:
                changed = True
                refresh_needed = refresh_needed or level2_refresh
                active_entries = [e for e in state["entries"] if not e.get("deleted")]
                cleanup_candidates = [e for e in active_entries if e.get("level", 0) < 3]

        if max_paragraphs > 0 and len(active_entries) > max_paragraphs:
            # 如果还超限，删除最旧的每周摘要 (Level 1)
            weekly_entries = sorted(
                [e for e in cleanup_candidates if e.get("level") == 1],
                key=lambda e: e.get("created_at", 0),
            )
            if weekly_entries:
                extra = len(active_entries) - max_paragraphs
                extra = max(1, extra)
                to_delete = weekly_entries[:extra]
                if await self._delete_entries(to_delete):
                    changed = True
                    refresh_needed = True
                    # 重新计算状态
                    active_entries = [e for e in state["entries"] if not e.get("deleted")]
                    cleanup_candidates = [e for e in active_entries if e.get("level", 0) < 3]

        # 如果节点数仍然超限，继续删除直到降到限制以下
        if nodes_exceeded and max_nodes > 0:
            iteration = 0
            while nodes_exceeded and cleanup_candidates and iteration < MAX_CAPACITY_ITERATIONS:
                iteration += 1
                # 优先删除每日摘要
                daily_entries = sorted(
                    [e for e in cleanup_candidates if e.get("level") == 0],
                    key=lambda e: e.get("created_at", 0),
                )
                if daily_entries:
                    to_delete = daily_entries[:min(MAX_DELETE_BATCH_SIZE, len(daily_entries))]
                else:
                    # 如果没有每日摘要，删除每周摘要
                    weekly_entries = sorted(
                        [e for e in cleanup_candidates if e.get("level") == 1],
                        key=lambda e: e.get("created_at", 0),
                    )
                    if weekly_entries:
                        to_delete = weekly_entries[:min(7, len(weekly_entries))]
                    else:
                        break
                
                if await self._delete_entries(to_delete):
                    changed = True
                    refresh_needed = True
                    # 重新计算状态
                    active_entries = [e for e in state["entries"] if not e.get("deleted")]
                    cleanup_candidates = [e for e in active_entries if e.get("level", 0) < 3]
                    # 重新检查节点数
                    node_count = await self._get_node_count()
                    nodes_exceeded = node_count > max_nodes
                else:
                    break
            
            if nodes_exceeded:
                logger.warning("KG 节点数 %s 超过限制 %s，已尝试删除但仍超限。", node_count, max_nodes)

        return changed, refresh_needed

    async def _maybe_build_level2(self, state: dict, active_entries: list[dict]) -> tuple[bool, bool]:
        """尝试将 Level 1 (每周摘要) 压缩为 Level 2。"""
        finalized = state.get("finalized_level2", {})
        weekly_by_chat: dict[str, list[dict]] = {}
        for entry in active_entries:
            # 仅压缩 Level 1
            if entry.get("level") != 1:
                continue
            chat_id = entry.get("chat_id")
            if not chat_id or finalized.get(chat_id):
                continue
            weekly_by_chat.setdefault(chat_id, []).append(entry)

        if not weekly_by_chat:
            return False, False

        selected_chat = None
        selected_entries = None
        oldest_time = None
        for chat_id, entries in weekly_by_chat.items():
            entries_sorted = sorted(entries, key=lambda e: e.get("created_at", 0))
            if not entries_sorted:
                continue
            entry_time = entries_sorted[0].get("created_at", 0)
            if oldest_time is None or entry_time < oldest_time:
                oldest_time = entry_time
                selected_chat = chat_id
                selected_entries = entries_sorted

        if not selected_chat or not selected_entries or len(selected_entries) < 2:
            return False, False

        tz = self._get_timezone()
        max_input_chars = int(self.get_config("summary.max_input_chars", 8000))
        level2_max_chars = int(self.get_config("summary.level2_max_chars", 1600))
        auto_import = bool(self.get_config("summary.auto_import", True))
        entries_text = await self._build_entries_text(selected_entries, tz)
        entries_text = self._truncate_text(entries_text, max_input_chars)
        summary = await self._summarize_text(entries_text, level2_max_chars, "level2")
        if not summary:
            return False, False

        start_ts = float(selected_entries[0].get("source_start") or selected_entries[0].get("created_at") or 0)
        end_ts = float(selected_entries[-1].get("source_end") or selected_entries[-1].get("created_at") or 0)
        if not end_ts:
            end_ts = _now_ts()

        entry = await self._persist_summary(
            chat_id=selected_chat,
            kind="level2",
            level=2,
            summary=summary,
            start_ts=start_ts,
            end_ts=end_ts,
            max_chars=level2_max_chars,
            tz=tz,
            import_flag=auto_import,
        )
        if not entry:
            return False, False
        state["entries"].append(entry)
        state.setdefault("finalized_level2", {})[selected_chat] = True
        await self._delete_entries(selected_entries)
        return True, True

    # ===== 待确认导入任务管理方法 =====

    async def _add_pending_confirm_task(self, task: PendingConfirmTask) -> None:
        """添加待确认导入任务"""
        async with self._pending_tasks_lock:
            self._pending_confirm_tasks[task.task_id] = task.to_dict()
            logger.info("添加待确认导入任务: %s (%s)", task.task_id, task.task_type)

    async def _get_pending_confirm_task(self, task_id: str) -> PendingConfirmTask | None:
        """获取待确认导入任务"""
        async with self._pending_tasks_lock:
            task_data = self._pending_confirm_tasks.get(task_id)
            if task_data:
                return PendingConfirmTask.from_dict(task_data)
            return None

    async def _remove_pending_confirm_task(self, task_id: str) -> PendingConfirmTask | None:
        """移除待确认导入任务"""
        async with self._pending_tasks_lock:
            task_data = self._pending_confirm_tasks.pop(task_id, None)
            if task_data:
                task = PendingConfirmTask.from_dict(task_data)
                logger.info("移除待确认导入任务: %s (%s)", task.task_id, task.task_type)
                return task
            return None

    async def _cleanup_expired_confirm_tasks(self) -> list[PendingConfirmTask]:
        """清理过期的待确认任务，返回被清理的任务列表"""
        current_time = time.time()
        expired_tasks = []

        async with self._pending_tasks_lock:
            task_ids_to_remove = []
            for task_id, task_data in self._pending_confirm_tasks.items():
                task = PendingConfirmTask.from_dict(task_data)
                if task.is_expired(current_time):
                    task_ids_to_remove.append(task_id)
                    expired_tasks.append(task)

            for task_id in task_ids_to_remove:
                self._pending_confirm_tasks.pop(task_id, None)
                logger.info("清理过期待确认导入任务: %s", task_id)

        # 对过期任务执行默认行为（拒绝导入，清理文件）
        for task in expired_tasks:
            try:
                await self._cleanup_task_files(task)
                logger.info("过期任务 %s 已自动拒绝并清理文件", task.task_id)
            except Exception as e:
                logger.error("清理过期任务文件失败: %s", e)

        return expired_tasks

    async def handle_confirm_decision(self, task_id: str, approved: bool) -> tuple[bool, str]:
        """处理管理员的确认决策"""
        try:
            task = await self._get_pending_confirm_task(task_id)
            if not task:
                return False, f"未找到待确认任务: {task_id}"

            # 移除待确认任务
            await self._remove_pending_confirm_task(task_id)

            if approved:
                # 执行导入
                ok = await self._import_raw_summary(task.raw_path, task.openie_path)
                if ok:
                    # 导入成功，创建状态条目
                    state = await self._load_state()
                    summary_hash = self._calculate_summary_hash(task.summary_text, task.chat_id, task.created_at)

                    # 构建状态条目（简化版）
                    entry = {
                        "id": f"{task.task_type}-{task.chat_id.replace('/', '_')}-{int(task.created_at)}-{summary_hash[:8]}",
                        "chat_id": task.chat_id,
                        "kind": task.task_type,
                        "level": 0,
                        "hash": summary_hash,
                        "raw_file": self._relpath(task.raw_path),
                        "openie_file": self._relpath(task.openie_path),
                        "created_at": task.created_at,
                        "source_start": task.created_at - 86400,  # 简化处理
                        "source_end": task.created_at,
                        "deleted": False,
                    }
                    state["entries"].append(entry)
                    await self._save_state(state)

                    return True, f"✅ 已确认导入任务 {task_id}，并成功导入到知识库"
                else:
                    # 导入失败，清理文件
                    await self._cleanup_task_files(task)
                    return False, f"❌ 任务 {task_id} 导入失败，已清理文件"
            else:
                # 拒绝导入，清理文件
                await self._cleanup_task_files(task)
                return True, f"❌ 已拒绝导入任务 {task_id}，已清理文件"

        except Exception as e:
            logger.error("处理确认决策失败: %s", e, exc_info=True)
            return False, f"处理确认决策时发生错误: {str(e)}"

    async def _cleanup_task_files(self, task: PendingConfirmTask) -> None:
        """清理任务相关的文件"""
        try:
            def _cleanup():
                if task.raw_path.exists():
                    task.raw_path.unlink()
                if task.openie_path.exists():
                    task.openie_path.unlink()

            await asyncio.to_thread(_cleanup)
            logger.debug("已清理任务文件: %s", task.task_id)
        except Exception as e:
            logger.warning("清理任务文件失败: %s", e)

    def _calculate_summary_hash(self, summary_text: str, chat_id: str, created_at: float) -> str:
        """计算摘要的哈希值"""
        try:
            from src.chat.knowledge.utils.hash import get_sha256
        except Exception:
            import hashlib
            def get_sha256(text: str) -> str:
                return hashlib.sha256(text.encode("utf-8")).hexdigest()

        # 构建与_persist_summary相同的文本格式
        meta = f"聊天会话 {chat_id} (确认导入)"
        full_text = f"{meta}\n\n{summary_text}"
        return get_sha256(full_text)

    async def _send_confirm_message(self, task: PendingConfirmTask) -> None:
        """发送确认导入消息给管理员"""
        try:
            # 获取确认消息的目标流
            confirm_stream = self.get_config("summary.confirm_stream", "").strip()
            if not confirm_stream:
                confirm_stream = self.get_config("performance.admin_id", "").strip()

            if not confirm_stream:
                logger.warning("未配置确认消息目标流，跳过发送确认消息")
                return

            # 生成简短的记忆总结
            short_summary = self._generate_short_summary(task.summary_text)

            # 构建确认消息
            task_type_name = {
                "daily": "每日摘要",
                "weekly": "每周摘要",
                "forever": "永久记忆"
            }.get(task.task_type, task.task_type)

            message_lines = [
                f"📝 **新的{task_type_name}等待确认导入**",
                f"任务ID: `{task.task_id}`",
                f"聊天: `{task.chat_id}`",
                f"时间: {datetime.fromtimestamp(task.created_at).strftime('%Y-%m-%d %H:%M:%S')}",
                "",
                "📄 **简短总结**:",
                short_summary,
                "",
                "⚠️  请确认是否导入此记忆到知识库：",
                f"同意导入: `/memories confirm {task.task_id} yes`",
                f"拒绝导入: `/memories confirm {task.task_id} no`",
                "",
                f"⏰ 此确认将在 {int((task.confirm_timeout - task.created_at) / 60)} 分钟后自动过期"
            ]

            message = "\n".join(message_lines)

            # 发送消息
            await send_api.send_text_message(confirm_stream, message, storage_message=False)
            logger.info("已发送确认消息到 %s: %s", confirm_stream, task.task_id)

        except Exception as e:
            logger.error("发送确认消息失败: %s", e, exc_info=True)

    def _generate_short_summary(self, summary_text: str, max_length: int = 200) -> str:
        """生成简短的记忆总结"""
        if len(summary_text) <= max_length:
            return summary_text

        # 尝试在句子边界截断
        truncated = summary_text[:max_length]
        last_sentence_end = max(
            truncated.rfind('。'),
            truncated.rfind('！'),
            truncated.rfind('？'),
            truncated.rfind('. '),
            truncated.rfind('! '),
            truncated.rfind('? ')
        )

        if last_sentence_end > max_length * 0.7:  # 如果句子结束位置合理
            return summary_text[:last_sentence_end + 1] + "..."
        else:
            return truncated + "..."
