"""Shared reader failure type, re-exported by the public reader module."""


class ArenaReaderError(RuntimeError):
    """Raised when a screen state or required visible field is ambiguous."""

    def __init__(self, code: str, detail: str, *, retry_whole_read: bool = True) -> None:
        self.code = code
        self.detail = detail
        self.retry_whole_read = retry_whole_read
        super().__init__(f"{code}: {detail}")
