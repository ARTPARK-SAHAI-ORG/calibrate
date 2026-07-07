"""Parse Fern-generated reference.md and map SDK methods to public API routes."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MAP = REPO_ROOT / "scripts" / "public_api_sdk_map.json"

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

    @property
    def openapi_key(self) -> tuple[str, str]:
        return self.http.upper(), self.path

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
    signature: str
    description: str
    usage_code: str

    @property
    def sdk_key(self) -> tuple[str, str]:
        return self.sdk_group, self.sdk_method


def load_route_map(path: Path | None = None) -> list[SdkRoute]:
    map_path = path or DEFAULT_MAP
    data = json.loads(map_path.read_text(encoding="utf-8"))
    routes: list[SdkRoute] = []
    for entry in data["routes"]:
        routes.append(
            SdkRoute(
                http=entry["http"],
                path=entry["path"],
                sdk_group=entry["sdk_group"],
                sdk_method=entry["sdk_method"],
                api_group=entry["api_group"],
            )
        )
    return routes


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
        line = line.replace('api_key="<value>"', 'api_key="sk_your_api_key"')
        line = line.replace('api_key="YOUR_API_KEY"', 'api_key="sk_your_api_key"')
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
        signature_match = re.search(
            rf"client\.{re.escape(sdk_group)}\.{re.escape(sdk_method)}\([^)]*\)",
            block,
        )
        signature = signature_match.group(0) if signature_match else (
            f"client.{sdk_group}.{sdk_method}(...)"
        )
        doc = SdkMethodDoc(
            sdk_group=sdk_group,
            sdk_method=sdk_method,
            signature=signature,
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
