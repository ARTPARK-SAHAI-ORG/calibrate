"""Parse Fern-generated reference.md and map SDK methods to public API routes."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OPENAPI = REPO_ROOT / "docs" / "api-reference" / "openapi.json"
DEFAULT_FERN_OVERRIDE_CANDIDATES = (
    REPO_ROOT.parent / "pense-backend" / "fern" / "openapi-overrides.yml",
    REPO_ROOT.parent / "calibrate-backend" / "fern" / "openapi-overrides.yml",
)

METHOD_HEADER = re.compile(
    r"<details><summary><code>client\.(\w+)\.<a[^>]+>(\w+)</a>",
    re.IGNORECASE,
)
DETAILS_END = re.compile(r"</details>", re.IGNORECASE)
USAGE_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)
HTML_TAG = re.compile(r"<[^>]+>")


@dataclass(frozen=True)
class SdkRoute:
    http: str
    path: str
    sdk_group: str
    sdk_method: str
    api_group: str
    title: str

    @property
    def sdk_key(self) -> tuple[str, str]:
        return self.sdk_group, self.sdk_method

    @property
    def mintlify_api_page(self) -> str:
        return f"{self.http.upper()} {self.path}"

    @property
    def doc_slug(self) -> str:
        group_dir = self.sdk_group.replace("_", "-")
        return f"sdk/{group_dir}/{self.sdk_method}"


@dataclass(frozen=True)
class SdkMethodDoc:
    sdk_group: str
    sdk_method: str
    description: str
    usage_code: str

    @property
    def sdk_key(self) -> tuple[str, str]:
        return self.sdk_group, self.sdk_method


def api_group_from_tag(tag: str) -> str:
    parts = tag.replace("-", " ").split()
    if not parts:
        return ""
    if len(parts) == 1:
        return parts[0].capitalize()
    return parts[0].capitalize() + " " + " ".join(part.lower() for part in parts[1:])


def fern_overrides_path() -> Path:
    raw = os.getenv("FERN_OVERRIDES_PATH", "").strip()
    if raw:
        return Path(raw)
    for candidate in DEFAULT_FERN_OVERRIDE_CANDIDATES:
        if candidate.is_file():
            return candidate
    raise SystemExit(
        "FERN_OVERRIDES_PATH is required when calibrate-backend/fern/openapi-overrides.yml "
        "is not available locally."
    )


def parse_fern_overrides(path: Path) -> dict[str, Any]:
    import yaml

    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or "paths" not in data:
        raise ValueError(f"Invalid Fern overrides file: {path}")
    return data


def build_route_map(
    overrides: dict[str, Any],
    openapi: dict[str, Any],
) -> list[SdkRoute]:
    routes: list[SdkRoute] = []
    openapi_paths = openapi.get("paths", {})
    for path in sorted(overrides.get("paths", {})):
        methods = overrides["paths"][path]
        if not isinstance(methods, dict):
            continue
        for method in sorted(methods):
            override = methods[method]
            if method.startswith("x") or not isinstance(override, dict):
                continue
            sdk_group = override.get("x-fern-sdk-group-name")
            sdk_method = override.get("x-fern-sdk-method-name")
            if not sdk_group or not sdk_method:
                continue
            op = openapi_paths.get(path, {}).get(method.lower())
            if not isinstance(op, dict):
                raise ValueError(f"OpenAPI spec missing {method.upper()} {path}")
            tags = op.get("tags") or []
            if not tags:
                raise ValueError(f"OpenAPI op missing tags for {method.upper()} {path}")
            routes.append(
                SdkRoute(
                    http=method.upper(),
                    path=path,
                    sdk_group=sdk_group,
                    sdk_method=sdk_method,
                    api_group=api_group_from_tag(tags[0]),
                    title=(op.get("summary") or "").strip(),
                )
            )
    return routes


def load_route_map(
    openapi: dict[str, Any] | None = None,
    overrides_path: Path | None = None,
) -> list[SdkRoute]:
    spec = openapi
    if spec is None:
        if not DEFAULT_OPENAPI.is_file():
            raise SystemExit(
                "OpenAPI spec is required when docs/api-reference/openapi.json is missing."
            )
        import json

        spec = json.loads(DEFAULT_OPENAPI.read_text(encoding="utf-8"))
    overrides = parse_fern_overrides(overrides_path or fern_overrides_path())
    return build_route_map(overrides, spec)


def _strip_html(text: str) -> str:
    text = HTML_TAG.sub("", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def _extract_description(block: str) -> str:
    marker = "#### 📝 Description"
    usage_marker = "#### 🔌 Usage"
    if marker not in block:
        return ""
    start = block.index(marker) + len(marker)
    end = block.find(usage_marker, start)
    chunk = block[start:end] if end != -1 else block[start:]
    return _strip_html(chunk)


def _simplify_usage_code(code: str) -> str:
    lines: list[str] = []
    for line in code.splitlines():
        if "CalibrateEnvironment" in line:
            continue
        if "environment=CalibrateEnvironment" in line:
            continue
        line = line.replace('api_key="<value>"', 'api_key="your_api_key"')
        line = line.replace('api_key="YOUR_API_KEY"', 'api_key="your_api_key"')
        lines.append(line)
    cleaned = "\n".join(lines).strip()
    while "\n\n\n" in cleaned:
        cleaned = cleaned.replace("\n\n\n", "\n\n")
    return cleaned + "\n"


def parse_reference_md(text: str) -> dict[tuple[str, str], SdkMethodDoc]:
    methods: dict[tuple[str, str], SdkMethodDoc] = {}
    for match in METHOD_HEADER.finditer(text):
        sdk_group = match.group(1)
        sdk_method = match.group(2)
        start = match.start()
        end_match = DETAILS_END.search(text, match.end())
        if not end_match:
            continue
        block = text[start : end_match.end()]
        usage_match = USAGE_BLOCK.search(block)
        if not usage_match:
            continue
        doc = SdkMethodDoc(
            sdk_group=sdk_group,
            sdk_method=sdk_method,
            description=_extract_description(block),
            usage_code=_simplify_usage_code(usage_match.group(1)),
        )
        methods[doc.sdk_key] = doc
    return methods


def parse_reference_file(path: Path) -> dict[tuple[str, str], SdkMethodDoc]:
    return parse_reference_md(path.read_text(encoding="utf-8"))


def routes_with_sdk_docs(
    routes: list[SdkRoute],
    methods: dict[tuple[str, str], SdkMethodDoc],
) -> list[tuple[SdkRoute, SdkMethodDoc]]:
    paired: list[tuple[SdkRoute, SdkMethodDoc]] = []
    for route in routes:
        doc = methods.get(route.sdk_key)
        if doc is not None:
            paired.append((route, doc))
    return paired
