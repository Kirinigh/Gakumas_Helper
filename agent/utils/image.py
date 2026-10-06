import zlib
import struct


def encode_bgr_png(image) -> bytes | None:
    """将 BGR 格式的截图编码为 PNG 字节（纯标准库实现，不依赖 Pillow/CV2）

    Args:
        image: numpy 数组，形状 (h, w, c>=3)，dtype uint8，通道顺序为 BGR

    Returns:
        bytes | None: PNG 字节；图像格式非法时返回 None
    """
    if image is None or getattr(image, "ndim", 0) != 3 or image.shape[2] < 3 or image.dtype != "uint8":
        return None

    height, width = image.shape[:2]
    if height == 0 or width == 0:  # 截图失败时 maa 返回 (0, 0, 3) 空数组
        return None

    # 每行前置过滤字节 0，同时把 BGR 翻转为 PNG 所需的 RGB（反转视图由 tobytes 按 C 序展开）
    raw = b"".join(b"\x00" + row.tobytes() for row in image[:, :, 2::-1])

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    # IHDR: 8 位深，颜色类型 2（真彩 RGB），无隔行
    ihdr = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    png = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")

    return png
