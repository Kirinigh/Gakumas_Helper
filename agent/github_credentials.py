"""Receive the desktop's active-profile credential once, without persisting it."""

import os

ENVIRONMENT_KEY = "MFA_GITHUB_TOKEN"
_token = ""


def consume_launch_token() -> None:
    global _token
    # Remove before pip/RIS workers are launched; their inherited environment
    # must not contain the desktop credential. No separate user setting.
    value = os.environ.pop(ENVIRONMENT_KEY, "").strip()
    _token = value if value and all(33 <= ord(char) <= 126 for char in value) else ""


def get_launch_token() -> str:
    return _token
