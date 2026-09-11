import os
import re
import csv
import sys
import json
import base64
import shutil
import hashlib
import subprocess
from typing import Optional
from pathlib import Path, PurePosixPath
from importlib import util, metadata, invalidate_caches

# 获取当前main.py所在路径并设置上级目录为工作目录
current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)
parent_dir = os.path.dirname(current_dir)
os.chdir(parent_dir)
# print(f"设置工作目录为: {parent_dir}")

# 将当前目录添加到路径
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

try:
    from utils import logger
except ImportError:
    # 如果logger不存在，创建一个简单的logger
    import logging

    logging.basicConfig(format="%(asctime)s | %(levelname)s | %(message)s", level=logging.INFO)
    logger = logging

from utils.win32_input_gate import enforce_maapicli_win32_input_gate


def _start_agent_server(agent_server, socket_id: str) -> None:
    evidence = enforce_maapicli_win32_input_gate()
    if evidence is not None:
        logger.info(
            "MaaPiCli Win32 输入权限屏障通过: "
            f"python_pid={evidence.python_pid}, maapicli_pid={evidence.maapicli_pid}, "
            f"game_pid={evidence.game_pid}, game_hwnd={evidence.game_window_handle}, "
            f"integrity_rid=0x{evidence.integrity_rid:04x}"
        )
    agent_server.start_up(socket_id)


def read_pip_config() -> dict:
    """
    读取 pip 配置文件并返回配置字典
    """
    config_dir = Path("./config")
    config_dir.mkdir(exist_ok=True)

    config_path = config_dir / "pip_config.json"
    default_config = {
        "enable_pip_update": True,
        "enable_pip_install": True,
        "last_version": "unknown",
        "mirror": "https://mirrors.ustc.edu.cn/pypi/simple",
        "backup_mirrors": [
            "https://pypi.tuna.tsinghua.edu.cn/simple",
            "https://mirrors.cloud.tencent.com/pypi/simple/",
            "https://pypi.org/simple",
        ],
    }

    if not config_path.exists():
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(default_config, f, indent=4)
        return default_config

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.exception("读取pip配置失败，使用默认配置")
        return default_config


def get_available_mirror(pip_config: dict | None) -> Optional[str]:
    """
    检查镜像源可用性并返回一个可用的镜像源
    """
    if pip_config is None:
        return None
    mirrors = [pip_config.get("mirror")] + pip_config.get("backup_mirrors", [])
    for mirror in mirrors:
        try:
            logger.info(f"尝试连接镜像源: {mirror}")
            response = subprocess.run(
                [sys.executable, "-m", "pip", "list", "-i", mirror],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
            )
            if response.returncode == 0:
                logger.info(f"镜像源可用: {mirror}")
                return mirror
        except Exception:
            logger.warning(f"镜像源不可用: {mirror}")
    logger.error("所有镜像源都不可用")
    return None


def install_requirements(req_file="requirements.txt", pip_config=None) -> bool:
    """
    安装 requirements.txt 中的依赖
    """
    req_path = Path(req_file)
    if not req_path.exists():
        logger.error(f"requirements.txt 不存在")
        return False

    return _install_pip_packages(["-r", str(req_path)], pip_config)


def _install_pip_packages(arguments: list[str], pip_config=None) -> bool:
    # 获取可用的镜像源
    mirror = get_available_mirror(pip_config)
    if not mirror:
        logger.error("没有可用的镜像源，安装依赖失败")
        return False

    try:
        logger.info("开始安装依赖...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-U",
            *arguments,
            "--no-warn-script-location",
            "-i",
            mirror,
        ]

        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if process.stdout:
            for line in process.stdout:
                line = line.strip()
                if not line:
                    continue
                if "Collecting" in line:
                    pkg = line.replace("Collecting", "").strip()
                    logger.info(f"正在安装: {pkg}")
                elif "Downloading" in line:
                    pkg = line.replace("Downloading", "").strip().split()[0]
                    logger.info(f"下载: {pkg}")
                elif "Installing collected packages" in line:
                    pkg = line.replace("Installing collected packages:", "").strip()
                    logger.info(f"安装完成: {pkg}")
        process.wait()
        if process.returncode == 0:
            logger.info("依赖安装完成")
            return True
        else:
            logger.error("依赖安装失败")
            return False
    except Exception as e:
        logger.exception("pip 安装依赖时出错")
        return False


def update_pip(pip_config=None):
    """
    更新 pip 到最新版本
    """
    mirror = get_available_mirror(pip_config)
    if not mirror:
        logger.error("没有可用的镜像源，无法更新 pip")
        return False

    try:
        logger.info("正在更新 pip...")
        cmd = [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--upgrade",
            "pip",
            "--no-warn-script-location",
            "-i",
            mirror,
        ]

        subprocess.check_call(cmd)
        logger.info("pip 更新成功")
        return True
    except Exception as e:
        logger.exception("更新 pip 时出错")
        return False


def read_required_maafw_version(req_file="requirements.txt") -> str:
    """以随包依赖声明为准；客户端升级时必须同步更新这一配套版本。"""
    requirements = Path(req_file).read_text(encoding="utf-8-sig").splitlines()
    versions = []
    for line in requirements:
        requirement = line.partition("#")[0].strip()
        if re.match(r"maafw(?:\s|[=<>!~;\[]|$)", requirement, flags=re.IGNORECASE):
            match = re.fullmatch(r"maafw\s*==\s*(\d+\.\d+\.\d+)", requirement, flags=re.IGNORECASE)
            if match is None:
                raise RuntimeError("随包 requirements.txt 必须明确指定配套的 maafw 版本")
            versions.append(match.group(1))
    if len(versions) != 1:
        raise RuntimeError("随包 requirements.txt 必须且只能声明一个配套的 maafw 版本")
    return versions[0]


def read_installed_maafw_version() -> str | None:
    # 不导入 maa，避免校正依赖前加载待替换的 MaaAgentServer.dll。
    try:
        return metadata.version("maafw")
    except metadata.PackageNotFoundError:
        return None


def reconcile_overlaid_maafw(required_version: str) -> str | None:
    """完整包覆盖会保留旧版本记录；仅在重复时核对目标 wheel 并清理旧记录。"""
    spec = util.find_spec("maa")
    if spec is None or spec.origin is None:
        return None
    site_root = Path(spec.origin).resolve().parent.parent
    records = list(site_root.glob("maafw-*.dist-info"))
    if len(records) < 2:
        return None
    target = site_root / f"maafw-{required_version}.dist-info"
    try:
        if target not in records:
            raise ValueError("覆盖包缺少目标版本记录")
        for directory in records:
            if directory.is_symlink() or directory.resolve().parent != site_root:
                raise ValueError("版本记录目录位置异常")
            distribution = metadata.PathDistribution(directory)
            if distribution.metadata.get("Name", "").lower() != "maafw":
                raise ValueError("版本记录的包名不一致")
            if directory == target and distribution.version != required_version:
                raise ValueError("目标版本记录不一致")
        seen = set()
        with (target / "RECORD").open(encoding="utf-8", newline="") as stream:
            for row in csv.reader(stream):
                if len(row) != 3:
                    raise ValueError("文件记录格式错误")
                name, digest, size = row
                relative = PurePosixPath(name)
                if (name in seen or "\\" in name or relative.is_absolute()
                        or ".." in relative.parts or not relative.parts
                        or relative.parts[0] not in {"maa", target.name}):
                    raise ValueError("文件记录路径错误或重复")
                seen.add(name)
                file = site_root.joinpath(*relative.parts)
                if not file.resolve().is_relative_to(site_root) or not file.is_file():
                    raise ValueError("覆盖包文件缺失")
                if name == f"{target.name}/RECORD":
                    if digest or size:
                        raise ValueError("文件记录自身应保留空摘要")
                    continue
                if not digest.startswith("sha256="):
                    raise ValueError("覆盖包缺少文件摘要")
                actual = base64.urlsafe_b64encode(hashlib.sha256(file.read_bytes()).digest()).decode().rstrip("=")
                if digest != f"sha256={actual}" or str(file.stat().st_size) != size:
                    raise ValueError("覆盖包文件与目标版本不一致")
        required = {"maa/__init__.py", "maa/define.py", f"{target.name}/METADATA", f"{target.name}/RECORD"}
        required.update(f"maa/bin/{name}.dll" for name in ("MaaFramework", "MaaAgentClient", "MaaAgentServer"))
        if not required.issubset(seen):
            raise ValueError("覆盖包记录不完整")
        # 不按旧 RECORD 卸载：它与新版共用 maa 文件，卸载会破坏已经覆盖的新版。
        for directory in records:
            if directory != target:
                shutil.rmtree(directory)
        invalidate_caches()
        logger.info(f"已整理覆盖更新遗留的 maafw 版本记录，当前 {required_version}")
        return required_version
    except (OSError, ValueError, AttributeError, KeyError) as error:
        raise RuntimeError(f"maafw 覆盖更新尚未完整应用：{error}") from error


def check_and_install_dependencies():
    """
    检查并安装依赖
    """
    required_maafw = read_required_maafw_version()
    overlay_error = None
    try:
        installed_maafw = reconcile_overlaid_maafw(required_maafw) or read_installed_maafw_version()
    except RuntimeError as error:
        overlay_error = error
        installed_maafw = None
    maafw_mismatch = installed_maafw != required_maafw
    pip_config = read_pip_config()
    enable_pip_update = pip_config.get("enable_pip_update", True)
    enable_pip_install = pip_config.get("enable_pip_install", True)

    if maafw_mismatch and not enable_pip_install:
        raise RuntimeError(
            (f"{overlay_error}。" if overlay_error else "") +
            f"maafw 配套版本不匹配：当前 {installed_maafw or '未安装'}，需要 {required_maafw}。"
            "已禁用依赖安装，请启用后重试，或恢复完整客户端包中的配套依赖。"
        )

    current_version = read_interface_version()
    last_version = pip_config.get("last_version", "unknown")

    logger.info(f"启用 pip 安装依赖: {enable_pip_install}")
    logger.info(f"当前版本: {current_version}, 上次运行版本: {last_version}")

    full_install = enable_pip_install and (current_version != last_version or current_version == "unknown")
    if not (full_install or maafw_mismatch):
        logger.info("无需安装依赖，跳过依赖安装与 pip 更新")
        return

    if enable_pip_update:
        if not update_pip(pip_config=pip_config):
            logger.warning("pip 更新失败，继续尝试安装依赖...")

    if overlay_error:
        logger.warning(f"{overlay_error}；重新安装配套依赖")
        installed = _install_pip_packages(["--force-reinstall", f"maafw=={required_maafw}"], pip_config)
        full_install = False
    elif full_install:
        installed = install_requirements(pip_config=pip_config)
    elif maafw_mismatch:
        logger.warning(f"校正 maafw 配套版本：{installed_maafw or '未安装'} → {required_maafw}")
        installed = _install_pip_packages([f"maafw=={required_maafw}"], pip_config)
    if not installed:
        raise RuntimeError("依赖安装失败，已停止启动；请检查网络后重试")
    actual_maafw = reconcile_overlaid_maafw(required_maafw) or read_installed_maafw_version()
    if actual_maafw != required_maafw:
        raise RuntimeError(
            f"依赖安装后 maafw 仍不匹配：当前 {actual_maafw or '未安装'}，需要 {required_maafw}；已停止启动"
        )
    if full_install and not update_pip_config(current_version):
        raise RuntimeError("依赖已安装，但无法保存依赖检查状态；请检查配置目录权限后重试")
    logger.info("依赖检查完成")


def read_interface_version(interface_file="./interface.json") -> str:
    """
    读取 interface.json 文件中的版本信息
    """
    interface_path = Path(interface_file)

    if not interface_path.exists():
        logger.warning("interface.json不存在")
        return "unknown"

    try:
        with open(interface_path, "r", encoding="utf-8") as f:
            interface_data = json.load(f)
            return interface_data.get("version", "unknown")
    except Exception as e:
        logger.exception("读取interface.json版本失败")
        return "unknown"


def update_pip_config(version) -> bool:
    """
    更新 pip 配置文件中的版本信息
    """
    config_path = Path("./config/pip_config.json")
    try:
        config = read_pip_config()
        config["last_version"] = version

        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=4)
        return True
    except Exception as e:
        logger.exception("更新pip配置失败")
        return False


def agent():
    try:
        import custom
        from utils import logger
        from maa.toolkit import Toolkit
        from arena_winrate import DEFAULT_OWN_SCORE_CACHE, OwnScoreCacheStore
        from arena_winrate.task_log import register_arena_log_sinks
        from maa.agent.agent_server import AgentServer
        from arena_winrate.upstream_component import sync_arena_season_ui_on_startup

        sync_arena_season_ui_on_startup()

        try:
            OwnScoreCacheStore(DEFAULT_OWN_SCORE_CACHE).ensure_summary()
            logger.info("竞技场己方缓存任务页摘要已同步")
        except Exception as e:
            logger.warning(f"竞技场己方缓存任务页摘要同步失败，下次启动将重试: {e}")

        Toolkit.init_option("./")

        socket_id = sys.argv[-1]

        arena_log_sinks = register_arena_log_sinks(AgentServer, logger)
        _start_agent_server(AgentServer, socket_id)
        logger.info("AgentServer 启动")
        AgentServer.join()
        AgentServer.shut_down()
        del arena_log_sinks
        logger.info("AgentServer 关闭")
    except Exception as e:
        logger.exception("Agent 运行过程中发生异常")
        raise


def main():
    check_and_install_dependencies()
    agent()


if __name__ == "__main__":
    main()
