#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""Fail if any file in the repository carries a credential.

Scans every text file under the repository root for the shapes a leaked secret takes:
bearer tokens and API keys, private-key blocks, passwords assigned to a
variable, SSH/SFTP transport code, the serving endpoint's host name, the remote
account name, bare IPv4 addresses, and any of the 2026 campaigns'
environment variables with a value attached.

Hits are split in two.  A *hard* hit exits non-zero: it is the shape a real
credential takes and a human has to look at it.  A *soft* hit is tolerated,
because what it matches is a name rather than a value.  Pass ``--verbose`` to
list soft hits for audit.

Run:  python code/check_no_secrets.py
"""
import io
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

SKIP_DIRS = {".git", "__pycache__", "_baseline_snapshot", "_work"}
SKIP_EXT = {".pdf", ".png", ".jpg", ".jpeg", ".gif", ".zip", ".pyc",
            ".aux", ".log", ".out", ".synctex", ".fls", ".fdb_latexmk"}

# files that could execute or hold configuration; prose is judged more
# leniently, but only on the rules where a mention is not a disclosure
CODE_EXT = {".py", ".sh", ".bash", ".ps1", ".json", ".yaml", ".yml",
            ".cfg", ".ini", ".env", ".toml", ".tex", ".bib"}

# (name, pattern, hard?, scope)  scope "all" | "code"
RULES = [
    ("openai-style key", r"sk-[A-Za-z0-9_\-]{16,}", True, "all"),
    ("bearer token", r"[Bb]earer\s+[A-Za-z0-9._\-]{16,}", True, "all"),
    ("private key block", r"-----BEGIN [A-Z ]*PRIVATE KEY-----", True, "all"),
    ("assigned password", r"(?i)\b(pass(word|wd)?|pwd)\b\s*[:=]\s*[\"'][^\"']{3,}", True, "all"),
    ("assigned api key", r"(?i)\bapi[_\-]?key\b\s*[:=]\s*[\"'][^\"']{8,}", True, "all"),
    ("long hex/base64 literal", r"[\"'][A-Za-z0-9+/]{40,}={0,2}[\"']", True, "all"),
    ("SSH/SFTP transport", r"(?i)\b(paramiko|SSHClient|sftp|ssh\s+-|scp\s+)", True, "code"),
    ("serving endpoint host", r"(?i)REDACTED-ENDPOINT", False, "all"),
    ("remote account", r"(?i)\banonymous-user\b", True, "all"),
    ("IPv4 literal", r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d.])", True, "all"),
    ("assigned campaign var",
     r"(?:LLM_API_KEYS?|LLM_BASE_URL|OPENAI_API_KEY)\s*[:=]\s*[^\s{}]{6,}", True, "all"),
    ("campaign var named", r"LLM_API_KEYS?|EXTRA_BODY_JSON|LLM_BASE_URL", False, "all"),
    ("env read", r"os\.environ|os\.getenv", False, "code"),
]
COMPILED = [(n, re.compile(p), hard, scope)
            for n, p, hard, scope in RULES]

hits = []
scanned = 0
VERBOSE = "--verbose" in sys.argv[1:]
for dirpath, dirnames, filenames in os.walk(ROOT):
    dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
    for fn in filenames:
        if os.path.splitext(fn)[1].lower() in SKIP_EXT:
            continue
        p = os.path.join(dirpath, fn)
        if os.path.abspath(p) == os.path.abspath(__file__):
            continue  # the scanner necessarily contains the patterns it looks for
        try:
            text = io.open(p, encoding="utf-8", errors="strict").read()
        except (UnicodeDecodeError, OSError):
            continue
        scanned += 1
        rel = os.path.relpath(p, ROOT)
        is_code = os.path.splitext(fn)[1].lower() in CODE_EXT
        for line_no, line in enumerate(text.splitlines(), 1):
            for name, rx, hard, scope in COMPILED:
                if scope == "code" and not is_code:
                    continue
                m = rx.search(line)
                if m and name == "long hex/base64 literal" and re.fullmatch(r"[\"'](?:[0-9a-f]{40}|[0-9a-f]{64})[\"']", m.group(0)):
                    continue  # a SHA-1/SHA-256 digest (lock and source hashes) is a fingerprint, not a credential
                if m and name == "IPv4 literal" and (m.group(0).startswith("127.") or line[max(0, m.start() - 2):m.start()] == "=="):
                    continue  # the loopback address and four-part version pins (pkg==12.4.5.8) are not hosts
                if m and name == "assigned campaign var" and re.search(r"os\.(environ|getenv)", line[m.start():]):
                    continue  # reading the variable from the environment discloses nothing
                if m and name == "assigned campaign var" and re.match(r"\w+\s*[:=]\s*EMPTY\b", m.group(0)):
                    continue  # EMPTY is the placeholder key of the local vLLM server, which checks no key
                if m:
                    hits.append((hard, name, rel, line_no, line.strip()[:110]))

hard_hits = [h for h in hits if h[0]]
soft_hits = [h for h in hits if not h[0]]

if hard_hits:
    print("scanned %d text files under %s\n" % (scanned, ROOT))
    print("== must be reviewed before release ==")
    for _h, name, rel, ln, line in hard_hits:
        print("  %-22s %s:%d\n      %s" % (name, rel, ln, line))
elif VERBOSE:
    print("scanned %d text files under %s\n" % (scanned, ROOT))
    print("== no credential-shaped string found ==")

if soft_hits and VERBOSE:
    print("\n== expected, listed for completeness ==")
    for _h, name, rel, ln, line in soft_hits:
        print("  %-22s %s:%d\n      %s" % (name, rel, ln, line))

sys.exit(1 if hard_hits else 0)
