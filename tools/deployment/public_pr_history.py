"""Fixed Git objects for the independently verified public pull request."""
from __future__ import annotations

import hashlib
from collections.abc import Mapping, Callable

REPOSITORY = "https://github.com/Kirinigh/Gakumas_Helper"
PR_NUMBER = 3
PR_CONTRIBUTOR = "NanoAstral"
PR_BASE = "88e929059f65d85efe00c3f63c751782d8383f47"
PR_COMMITS = (
    "7e8473f6f0b0e05ebef24971b4fabffd79922776",
    "ef5f1a14f238fa7b6dd0b170ce0b4cccfa6ad0f2",
)
PR_MERGE = "16f479ad0dd514036096dc32641ea092010b7af1"
PR_PARENTS = {
    PR_COMMITS[0]: (PR_BASE,),
    PR_COMMITS[1]: (PR_COMMITS[0],),
    PR_MERGE: (PR_BASE, PR_COMMITS[1]),
}


def verify_public_pr(
    parents: Mapping[str, tuple[str, ...]],
    *,
    repository: str,
    read_raw_commit: Callable[[str], bytes],
) -> frozenset[str]:
    """Allow the full previously published PR, with its original Git bytes."""
    found = PR_PARENTS.keys() & parents.keys()
    if not found:
        return frozenset()
    if repository != REPOSITORY or found != PR_PARENTS.keys():
        raise ValueError("Public PR history has an incomplete or foreign adoption")
    for commit, expected_parents in PR_PARENTS.items():
        if parents[commit] != expected_parents:
            raise ValueError(f"Public PR ancestry differs from the verified adoption: {commit}")
        raw = read_raw_commit(commit)
        header = f"commit {len(raw)}\0".encode("ascii")
        if hashlib.sha1(header + raw).hexdigest() != commit:
            raise ValueError(f"Public PR commit object differs from the verified adoption: {commit}")
    return frozenset(found)
