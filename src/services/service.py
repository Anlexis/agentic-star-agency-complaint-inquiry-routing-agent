"""AgentCore Platform v1.0"""

# Shared, side-effect-free validators for the complaint routing pipeline.
#
# Everything here answers one of three questions, and every caller — the HTTP
# adapter, the input node, the output gate — asks them through this module so a
# single definition cannot drift into three near-copies:
#
#   1. Is this caller-supplied value finite, bounded and safe to compute with?
#   2. Does this text carry an instruction aimed at a language model?
#   3. Does this text carry something credential-shaped?
#
# The module holds no business logic, performs no I/O, and constructs no client.

from __future__ import annotations

import math
import re
import unicodedata
import urllib.parse
from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from framework.security.credential_detector import detect_credentials

# ---------------------------------------------------------------------------
# Inert identifier alphabet
# ---------------------------------------------------------------------------
#
# Caller-supplied identifiers are rendered verbatim into the routing record, so
# they are restricted to an alphabet that cannot carry punctuation, markup, a
# newline, or a directive. Free text is never rendered.

_INERT_RE = re.compile(r"^[a-z0-9_]{1,32}$")

# The session identifier travels into correlation and audit records rather than
# into the rendered decision, so it admits the wider conventional alphabet.
_SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def is_inert_token(value: object) -> bool:
    """True when *value* is a 1-32 character ``[a-z0-9_]`` identifier."""
    return isinstance(value, str) and bool(_INERT_RE.match(value))


def is_session_token(value: object) -> bool:
    """True when *value* is a 1-64 character ``[A-Za-z0-9_-]`` identifier."""
    return isinstance(value, str) and bool(_SESSION_RE.match(value))


# ---------------------------------------------------------------------------
# Finite, bounded numbers
# ---------------------------------------------------------------------------


def finite_in_range(value: object, low: float, high: float) -> Optional[float]:
    """Return *value* as a finite float within ``[low, high]``, else ``None``.

    Rejects, in this order and for these reasons:

    * ``bool`` — ``isinstance(True, int)`` is True in Python, so a JSON ``true``
      would otherwise be accepted as the number 1.
    * anything that is not a number or a numeric string.
    * ``NaN`` / ``±Infinity`` — these parse perfectly well through ``float()``
      and arrive intact through raw JSON, and every comparison against NaN is
      False. A threshold check written the obvious way therefore *passes* on
      NaN, which is a silent fail-OPEN on the exact decision this template
      exists to make.
    * magnitudes outside the declared range.

    The caller decides what a ``None`` means; nothing here clamps, because a
    clamped value is an invented one.
    """
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        candidate = float(value)
    elif isinstance(value, str):
        try:
            candidate = float(value.strip())
        except (TypeError, ValueError):
            return None
    else:
        return None
    if not math.isfinite(candidate):
        return None
    if candidate < low or candidate > high:
        return None
    return candidate


def bounded_int(value: object, low: int, high: int) -> Optional[int]:
    """Return *value* as an integer within ``[low, high]``, else ``None``.

    Built on :func:`finite_in_range`, so the non-finite and boolean rejections
    hold here too. A fractional value is rejected rather than truncated.
    """
    candidate = finite_in_range(value, float(low), float(high))
    if candidate is None:
        return None
    if candidate != int(candidate):
        return None
    return int(candidate)


# ---------------------------------------------------------------------------
# Instruction (prompt-injection) screening
# ---------------------------------------------------------------------------
#
# The framework's own detector is the floor, but two classes it does not close
# matter here and are screened locally:
#
#   * ``<<SYS>>`` — the framework blocks ``<|im_start|>`` and ``[INST]`` at high
#     confidence but scores ``<<SYS>>`` at none, which is exactly what would hide
#     it behind a "the framework covers this" assumption.
#   * any ``<|...|>`` control token, not only the two named ones.
#
# Both directions are screened: the raw text, and the text with markup removed.
# Stripping markup is not a defence on its own — it can convert a detectable
# token attack into undetectable plain text, and it can re-assemble a directive
# that was split by tags (``ig<b>nore previous instructions``). Screening only
# one of the two forms therefore misses one of the two attacks.

_CONTROL_TOKEN_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(r"<\|[^|<>]{1,64}\|>"),  # <|im_start|>, <|endoftext|>
    re.compile(r"\[/?(?:INST|SYS|SYSTEM)\]", re.IGNORECASE),  # [INST] [/INST] [SYS]
    re.compile(r"<</?SYS>>", re.IGNORECASE),  # <<SYS>> <</SYS>>
    re.compile(r"</?\s*(?:system|assistant|user)\s*>", re.IGNORECASE),
    re.compile(r"###\s*(?:system|instruction)\b", re.IGNORECASE),
    re.compile(r"^\s*(?:system|assistant)\s*:", re.IGNORECASE | re.MULTILINE),
)

_DIRECTIVE_RES: Tuple[re.Pattern[str], ...] = (
    re.compile(
        r"(?:ignore|disregard|forget|override)\s+(?:all\s+|any\s+|the\s+)?"
        r"(?:previous|above|prior|earlier|system)\s+"
        r"(?:instructions?|prompts?|rules?|context)",
        re.IGNORECASE,
    ),
    re.compile(
        r"\bact\s+as\s+(?:a|an)\s+(?:different|new|unrestricted|unfiltered|evil|"
        r"jailbroken|dan|god|admin|root|superuser|hacker)\b",
        re.IGNORECASE,
    ),
    re.compile(r"\bjailbreak(?:ing)?\b", re.IGNORECASE),
    re.compile(r"\byour\s+(?:new\s+)?(?:system\s+)?(?:prompt|instructions?)\s+(?:is|are)\b", re.IGNORECASE),
    # Japanese equivalents — the complaint corpus is Japanese, so an
    # English-only screen would be a screen in the wrong language.
    #
    # Both forms REQUIRE the directive to address the system, because the
    # unanchored form fires on an ordinary complaint. "担当者が私の指示を無視した"
    # — the agent ignored my instruction — is a real conduct complaint, and it
    # is exactly the kind of complaint this template exists to route. A screen
    # that refuses it is not strict, it is broken in the direction that blocks
    # real work.
    re.compile(r"(?:これまでの|以前の|上記の|全ての|すべての)(?:指示|命令|ルール|プロンプト)を?(?:無視|忘れ)"),
    re.compile(r"(?:システム|AI|あなた)(?:への|の)?(?:指示|命令|ルール|プロンプト)を?(?:無視|忘れ)"),
    re.compile(r"システムプロンプト"),
)

_MARKUP_RE = re.compile(r"<[^<>]{0,200}>")
# Zero-width and soft-hyphen code points, written as escapes: a literal class of
# invisible characters is unreviewable and silently mutated by any editor that
# normalises whitespace.
_ZERO_WIDTH_RE = re.compile("[\u200b\u200c\u200d\ufeff\u00ad]")


def _normalize(text: str) -> str:
    """Fold the obfuscations that would otherwise walk straight past the patterns.

    URL-decode, NFKC-fold, strip zero-width characters — the same three steps
    the framework's own detector applies, so a payload that the framework would
    catch after normalisation is not one this screen misses before it.
    """
    folded = unicodedata.normalize("NFKC", urllib.parse.unquote(text))
    return _ZERO_WIDTH_RE.sub("", folded)


def strip_markup(text: str) -> str:
    """Remove tag-shaped spans so a directive split by markup re-assembles."""
    return _MARKUP_RE.sub("", text)


def detect_instructions(text: object) -> List[str]:
    """Return the closed-set labels of any model-directed instruction found.

    Screens the raw text AND the markup-stripped text: control tokens are
    caught before a strip could remove them, spliced directives after the strip
    re-assembles them. Labels are a fixed vocabulary, never the matched text —
    echoing a match back is how a rejected value ends up in a log.
    """
    if not isinstance(text, str) or not text:
        return []

    labels: Set[str] = set()
    for candidate in (_normalize(text), _normalize(strip_markup(text))):
        for pattern in _CONTROL_TOKEN_RES:
            if pattern.search(candidate):
                labels.add("control_token")
                break
        for pattern in _DIRECTIVE_RES:
            if pattern.search(candidate):
                labels.add("instruction_override")
                break
    return sorted(labels)


def screen_payload_for_instructions(payload: object, _depth: int = 0) -> List[str]:
    """Depth-first instruction screen over a parsed payload, KEYS included.

    Field names are caller data too, and a JSON ``\\u`` escape is already
    decoded by the time the payload is a Python object — so scanning after the
    parse, over keys as well as values, is what closes the escape route.
    """
    if _depth > 8:
        return ["structure_too_deep"]
    labels: Set[str] = set()
    if isinstance(payload, str):
        labels.update(detect_instructions(payload))
    elif isinstance(payload, Mapping):
        for key, value in payload.items():
            labels.update(detect_instructions(str(key)))
            labels.update(screen_payload_for_instructions(value, _depth + 1))
    elif isinstance(payload, (list, tuple)):
        for item in payload:
            labels.update(screen_payload_for_instructions(item, _depth + 1))
    return sorted(labels)


# ---------------------------------------------------------------------------
# Credential screening
# ---------------------------------------------------------------------------
#
# The framework's detector is the FLOOR, not a replacement. Two failure modes,
# opposite in direction, both real:
#
#   * a local set NARROWER than the framework's lets a value through that the
#     framework then raises on, from inside the output gate — the wrapper turns
#     that into a bare error partial and discards whatever clearing the gate had
#     done. A detector gap is a containment bypass.
#   * "delegating" to the framework and DELETING the local patterns is the same
#     bug in reverse: the framework's patterns describe credential *formats* and
#     match nothing of the ``password=...`` shape, so the swap looks like a
#     tightening and is a widening of what gets through.
#
# So: take the union. The locals below are the patterns the framework does not
# carry, and they are listed rather than summarised so a reviewer can see the
# delta.

_LOCAL_CREDENTIAL_RES: Tuple[Tuple[str, re.Pattern[str]], ...] = (
    (
        "assigned_secret",
        re.compile(
            r"\b(?:password|passwd|pwd|secret|api[_\-]?key|access[_\-]?key|client[_\-]?secret|auth[_\-]?token|token)"
            r"\s*[:=]\s*\S{4,}",
            re.IGNORECASE,
        ),
    ),
    ("private_key_block", re.compile(r"-----BEGIN [A-Z ]{0,32}PRIVATE KEY-----")),
    ("google_api_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("github_pat", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}")),
    ("gitlab_pat", re.compile(r"\bglpat-[A-Za-z0-9_\-]{16,}")),
    ("slack_token", re.compile(r"\bxox[abposr]-[A-Za-z0-9\-]{10,}")),
)


def detect_output_credentials(text: object) -> List[str]:
    """Return closed-set credential-type labels found in *text*.

    Union of the framework's ``detect_credentials`` and the local patterns
    above. Returns labels only — never the matched value, never an offset.
    """
    if not isinstance(text, str) or not text:
        return []
    labels: Set[str] = {str(finding["type"]) for finding in detect_credentials(text)}
    for label, pattern in _LOCAL_CREDENTIAL_RES:
        if pattern.search(text):
            labels.add(label)
    return sorted(labels)


def detect_credentials_in_structure(value: object, _depth: int = 0) -> List[str]:
    """Union credential screen over every string leaf of a JSON-like structure.

    Deliberately mirrors ``detect_credentials_in_value``: values only, never
    keys. That is not an oversight to be improved on locally — the framework's
    ``@final`` output gate scans values only, and a screen that diverges from
    the gate it is protecting is a screen whose refusal set no longer matches
    the block set.
    """
    if _depth > 8:
        return []
    if isinstance(value, str):
        return detect_output_credentials(value)
    labels: Set[str] = set()
    if isinstance(value, Mapping):
        for nested in value.values():
            labels.update(detect_credentials_in_structure(nested, _depth + 1))
    elif isinstance(value, (list, tuple)):
        for item in value:
            labels.update(detect_credentials_in_structure(item, _depth + 1))
    return sorted(labels)


def first_credential_field(context: Mapping[str, Any]) -> Optional[str]:
    """Return the name of the first context field carrying a credential shape.

    Iterating top-level fields is exactly equivalent to scanning the whole
    mapping, because the structure scan is defined as the union over its
    values. That identity is what lets the refusal name a field without either
    widening or narrowing what is refused.
    """
    for name in sorted(context):
        if detect_credentials_in_structure(context[name]):
            return name
    return None


# ---------------------------------------------------------------------------
# Redaction sentinel
# ---------------------------------------------------------------------------

_MASK_SENTINEL = "[MASKED]"
_SENTINEL_ONLY_RE = re.compile(r"^(?:\s*\[MASKED\]\s*)+$")


def is_redaction_sentinel(text: object) -> bool:
    """True when *text* is nothing but redaction sentinels.

    The platform masks personal-data shapes before any template code runs, so
    ``[MASKED]`` arrives as an ordinary string. Reporting a confident
    classification of a value that is only a sentinel would be certifying the
    redaction, not the complaint.
    """
    return isinstance(text, str) and bool(text) and bool(_SENTINEL_ONLY_RE.match(text))


def sentinel_ratio(text: object) -> float:
    """Fraction of *text* occupied by redaction sentinels, 0.0 when empty."""
    if not isinstance(text, str) or not text:
        return 0.0
    masked_chars = text.count(_MASK_SENTINEL) * len(_MASK_SENTINEL)
    return masked_chars / len(text)


# ---------------------------------------------------------------------------
# Bounded configuration
# ---------------------------------------------------------------------------


def bounded_config_int(
    config: Optional[Mapping[str, Any]],
    key: str,
    default: int,
    low: int,
    high: int,
) -> int:
    """Read ``config[key]`` as an integer in ``[low, high]``, else *default*.

    A declared value outside its range is DROPPED, not clamped: clamping
    invents a value the operator never wrote, and hides the mistake. Falling
    back to the documented default is visible and reversible.
    """
    if not isinstance(config, Mapping):
        return default
    resolved = bounded_int(config.get(key), low, high)
    return default if resolved is None else resolved


def config_section(config: Optional[Mapping[str, Any]], key: str) -> Dict[str, Any]:
    """Return ``config[key]`` when it is a mapping, else an empty mapping."""
    if isinstance(config, Mapping):
        section = config.get(key)
        if isinstance(section, Mapping):
            return dict(section)
    return {}
