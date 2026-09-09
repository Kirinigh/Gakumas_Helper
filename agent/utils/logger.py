import sys
from pathlib import Path
from datetime import timedelta

DEFAULT_LOG_DIR = Path(".local/runtime-data/logs")
LOG_ROTATION = "00:00"
LOG_RETENTION_DAYS = 14
LOG_COMPRESSION = "zip"

try:
    from loguru import logger as _logger


    def setup_logger(log_dir=DEFAULT_LOG_DIR, console_level="INFO"):
        """设置 loguru logger

        Args:
            log_dir: 日志文件目录
            console_level: 控制台输出等级 (DEBUG, INFO, WARNING, ERROR)
        """
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        _logger.remove()

        # 定义日志级别的简短格式
        def format_level(record):
            level_map = {
                "INFO": "info",
                "ERROR": "err",
                "WARNING": "warn",
                "DEBUG": "debug",
                "CRITICAL": "critical",
                "SUCCESS": "success",
                "TRACE": "trace",
            }
            record["extra"]["level_short"] = level_map.get(
                record["level"].name, record["level"].name.lower()
            )
            return record["extra"].get("ui_visible", True)

        _logger.add(
            sys.stderr,
            format="<level>{extra[level_short]}</level>:<level>{message}</level>",
            colorize=True,
            level=console_level,
            filter=format_level,
        )
        _logger.add(
            str(log_dir / "{time:YYYY-MM-DD}.log"),
            rotation=LOG_ROTATION,
            retention=timedelta(days=LOG_RETENTION_DAYS),
            compression=LOG_COMPRESSION,
            level="DEBUG",
            format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} | {message}",
            encoding="utf-8",
            enqueue=True,
            backtrace=True,  # 包含完整的异常回溯信息
            diagnose=False,  # 避免异常诊断把局部变量（可能含秘密）写入日志
        )
        return _logger


    def change_console_level(level="DEBUG"):
        """动态修改控制台日志等级"""
        setup_logger(console_level=level)
        _logger.info(f"控制台日志等级已更改为: {level}")


    logger = setup_logger()
except ImportError:
    import logging
    import zipfile
    from logging.handlers import TimedRotatingFileHandler


    class ShortLevelFormatter(logging.Formatter):
        """自定义 Formatter，使用简短的日志级别名称"""

        level_map = {
            "INFO": "info",
            "ERROR": "err",
            "WARNING": "warn",
            "DEBUG": "debug",
            "CRITICAL": "critical",
        }

        def format(self, record):
            record.level_short = self.level_map.get(
                record.levelname, record.levelname.lower()
            )
            return super().format(record)


    class CompatibleLogger:
        """Expose the subset of loguru methods used by the agent."""

        def __init__(self, wrapped):
            self._wrapped = wrapped

        def __getattr__(self, name):
            return getattr(self._wrapped, name)

        def success(self, message, *args, **kwargs):
            return self._wrapped.info(message, *args, **kwargs)

        def trace(self, message, *args, **kwargs):
            return self._wrapped.debug(message, *args, **kwargs)

        def bind(self, **extra):
            return CompatibleLogger(logging.LoggerAdapter(self._wrapped, extra))


    _fallback_logger = logging.getLogger("maagakumasu")


    def _zip_rotator(source, destination):
        with zipfile.ZipFile(destination, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.write(source, arcname=Path(source).name)
        Path(source).unlink(missing_ok=True)


    def setup_logger(log_dir=DEFAULT_LOG_DIR, console_level="INFO"):
        """Configure the dependency-free fallback with the same 14-day policy."""

        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        for old_handler in _fallback_logger.handlers[:]:
            old_handler.close()
            _fallback_logger.removeHandler(old_handler)

        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(console_level)
        console_handler.addFilter(lambda record: getattr(record, "ui_visible", True))
        console_handler.setFormatter(ShortLevelFormatter("%(level_short)s:%(message)s"))

        file_handler = TimedRotatingFileHandler(
            log_dir / "current.log",
            when="midnight",
            interval=1,
            backupCount=LOG_RETENTION_DAYS,
            encoding="utf-8",
            delay=True,
        )
        file_handler.setLevel(logging.DEBUG)
        file_handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d | %(message)s"))
        file_handler.namer = lambda name: f"{name}.zip"
        file_handler.rotator = _zip_rotator

        _fallback_logger.addHandler(console_handler)
        _fallback_logger.addHandler(file_handler)
        _fallback_logger.setLevel(logging.DEBUG)
        _fallback_logger.propagate = False
        return CompatibleLogger(_fallback_logger)


    def change_console_level(level="DEBUG"):
        global logger

        logger = setup_logger(console_level=level)
        logger.info(f"控制台日志等级已更改为: {level}")


    logger = setup_logger()
