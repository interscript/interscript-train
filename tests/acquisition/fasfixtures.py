"""Fixtures for the acquisition factory tests.

Builds a synthetic mini-corpus mirroring the real source structures:
a ganjoor-shaped REST API, a MediaWiki bz2 dump, a Common Crawl WET
segment set, and held-out eval jsonl files — all served from a local
HTTP server (no network). Also loads the FROZEN pre-refactor reference
scripts (tests/acquisition/reference, byte-verbatim from the base
commit) with only the source-base URLs pointed at the local server.
"""

from __future__ import annotations

import bz2
import copy
import gzip
import io
import json
import sys
import threading
import types
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REFERENCE_DIR = Path(__file__).resolve().parent / "reference"

# ---- codepoint palette ----
FATHATAN = "ً"
DAMMATAN = "ٌ"
KASRATAN = "ٍ"
FATHA = "َ"
DAMMA = "ُ"
KASRA = "ِ"
SHADDA = "ّ"
SUKUN = "ْ"
DAGGER_ALIF = "ٰ"
MADDAH_ABOVE = "ٓ"
HAMZA_ABOVE = "ٔ"
HAMZA_BELOW = "ٖ"
ZWNJ = "‌"

V_EVAL_SRC = "شاعر پرسید که یار دیرین کجاست امشب و سهراب"  # 42 chars; norm[0:40] = the held-out window


def vocalize(word: str, every: int = 3, mark: str = FATHA) -> str:
    out = []
    n = 0
    for c in word:
        out.append(c)
        if c != " " and c.isalpha():
            n += 1
            if n % every == 0:
                out.append(mark)
    return "".join(out)


# ---- ganjoor verse rows ----
V1_TGT = vocalize("الانسان در جهان تنها نیست")
V1_SRC = "الانسان در جهان تنها نیست"
V_UNVOC = "این سخن کوتاه است"
V_SHORT = "سلام"
V_LONG = "ط" * 310
V_RATIO_SRC = "کتاب دانش بود"
V_RATIO_TGT = "".join(
    (c + SHADDA + FATHA) if c != " " else c for c in V_RATIO_SRC)
V_LATIN = "hello world this is an english sentence ok"
V104_TGT = vocalize("باز هم دفتر شعر گشوده شد")
V104_SRC = "باز هم دفتر شعر گشوده شد"
V9_TGT = "تو" + HAMZA_ABOVE + "انست" + KASRA + "م که کار" + ZWNJ + "ستان است"
V9_SRC = "تو" + HAMZA_ABOVE + "انستم که کار" + ZWNJ + "ستان است"
V201_TGT = vocalize("چند پرسش ژرف", every=2)
V201_SRC = "چند پرسش ژرف"

GANJOOR_VERSES = {
    100: [
        {"vOrder": 1, "versePosition": "Right", "text": V1_TGT},
        {"vOrder": 2, "versePosition": "Left", "text": V_UNVOC},
    ],
    101: [
        {"vOrder": 1, "versePosition": "Right", "text": vocalize(V_SHORT)},
        {"vOrder": 2, "versePosition": "Left", "text": vocalize(V_LONG)},
        {"vOrder": 3, "versePosition": "Center", "text": V_RATIO_TGT},
        {"vOrder": 4, "versePosition": "Left", "text": V_LATIN},
    ],
    102: [
        {"vOrder": 1, "versePosition": "Right", "text": V1_TGT},
    ],
    103: [],
    104: [
        {"vOrder": 1, "versePosition": "Right", "text": V104_TGT},
    ],
    200: [
        {"vOrder": 1, "versePosition": "Right", "text": vocalize(V_EVAL_SRC)},
        {"vOrder": 2, "versePosition": "Left", "text": V9_TGT},
    ],
    201: [
        {"vOrder": 1, "versePosition": "Right", "text": V201_TGT},
        {"vOrder": 2, "versePosition": "Left", "text": ""},
    ],
}

GANJOOR_CATS = {
    10: {"poems": [{"id": 100, "title": "غزل یک"},
                   {"id": 101, "title": "غزل دو"},
                   {"id": 102, "title": "غزل سه"},
                   {"id": 103, "title": "غزل چهار"}],
         "children": [{"id": 20, "urlSlug": "ghazals"}]},
    20: {"poems": [{"id": 104, "title": "قطعه"}], "children": []},
    30: {"poems": [{"id": 200, "title": "بوستان"},
                   {"id": 201, "title": "گلستان"}],
         "children": []},
}

GANJOOR_POETS = [
    {"id": 1, "fullUrl": "/hafez", "name": "Hafez", "nickname": "حافظ",
     "rootCatId": 10, "published": True},
    {"id": 2, "fullUrl": "/saadi", "name": "Saadi", "nickname": "سعدی",
     "rootCatId": 30, "published": True},
    {"id": 3, "fullUrl": "/ghost", "name": "Ghost", "nickname": "غ",
     "rootCatId": 99, "published": False},
]

GANJOOR_TITLES = {100: "غزل یک", 101: "غزل دو", 102: "غزل سه",
                  104: "قطعه", 200: "بوستان", 201: "گلستان"}


def ganjoor_canonical_raw() -> dict[str, str]:
    """Deterministic stage-A raw files (hafez/saadi) in canonical poem
    order — the seed for the assemble-boundary byte-equivalence test."""
    rows: list[dict] = []

    def add(poem_id: int, cat_path: str, poet_url: str, poet_name: str):
        for v in GANJOOR_VERSES[poem_id]:
            tgt = v["text"]
            if not tgt:
                continue
            rows.append({
                "src": "".join(c for c in tgt
                               if not (0x064B <= ord(c) <= 0x0652 or ord(c) == 0x0670)).strip(),
                "tgt": tgt,
                "poet_url": poet_url,
                "poet_name": poet_name,
                "cat_path": cat_path,
                "poem_id": poem_id,
                "poem_title": GANJOOR_TITLES[poem_id],
                "vorder": v["vOrder"],
                "verse_position": v["versePosition"],
            })

    for pid in (100, 101, 102, 104):
        cat = "حافظ/ghazals" if pid == 104 else "حافظ"
        add(pid, cat, "/hafez", "Hafez")
    hafez = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)

    rows_b: list[dict] = []
    for pid in (200, 201):
        for v in GANJOOR_VERSES[pid]:
            tgt = v["text"]
            if not tgt:
                continue
            rows_b.append({
                "src": "".join(c for c in tgt
                               if not (0x064B <= ord(c) <= 0x0652 or ord(c) == 0x0670)).strip(),
                "tgt": tgt,
                "poet_url": "/saadi",
                "poet_name": "Saadi",
                "cat_path": "سعدی",
                "poem_id": pid,
                "poem_title": GANJOOR_TITLES[pid],
                "vorder": v["vOrder"],
                "verse_position": v["versePosition"],
            })
    saadi = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows_b)
    return {"hafez.jsonl": hafez, "saadi.jsonl": saadi}


# ---- wikisource dump XML ----

WIKI_PAGE_1_WT = (
    "== مقدمه ==\n"
    "این {{شعر|بیتِ نخست|بیتِ دوم}} است.\n"
    "[[حافظ|حافظِ شیراز]] می‌گوید.\n"
    "<ref>منبع</ref> <!-- پنهان --> {| جدول |}\n"
    "'''حافظ''' بهترین است.\n"
)
WIKI_PAGE_8_WT = (
    "نثر فارسی یکی از کهن‌ترین نثرهای ادبی است و در طول تاریخ "
    "گسترش بسیاری یافته است. کتاب‌های بسیاری به این زبان نوشته شده‌اند."
)
WIKI_PAGE_7_WT = V_EVAL_SRC + " و این متن ادامه دارد تا از پنجره گذرد."

WIKI_PAGES = [
    # (id, ns, title, wikitext)
    (1, "0", "دیوان حافظ", WIKI_PAGE_1_WT),
    (2, "0", "تغییرمسیر صفحه", "#تغییرمسیر [[دیوان حافظ]]"),
    (3, "2", "کاربر:نویسنده", "این یک صفحه کاربری است."),
    (4, "0", "کوتاه", "کم"),
    (5, "0", "انگلیسی",
     "This page is an English article about nothing in particular at all."),
    (6, "0", "دیوان حافظ", "محتوای تکراری."),
    (7, "0", "محذوف", WIKI_PAGE_7_WT),
    (8, "0", "نثر فارسی", WIKI_PAGE_8_WT),
]

MEDIA_NS = "http://www.mediawiki.org/xml/export-0.11/"


def wiki_dump_bz2() -> bytes:
    pages = []
    for pid, ns, title, wt in WIKI_PAGES:
        pages.append(
            f"<page><title>{title}</title><ns>{ns}</ns><id>{pid}</id>"
            f"<revision><id>{pid}0</id><text>{wt}</text></revision></page>")
    xml = (f'<mediawiki xmlns="{MEDIA_NS}" version="0.11">'
           "<siteinfo><sitename>wikifixture</sitename></siteinfo>"
           + "".join(pages) + "</mediawiki>")
    return bz2.compress(xml.encode("utf-8"))


# ---- Common Crawl WET fixture ----

def wet_record(warc_type: str, uri: str, lang: str | None,
               content: list[str]) -> str:
    lines = ["WARC/1.0", f"WARC-Type: {warc_type}",
             f"WARC-Target-URI: {uri}"]
    if lang is not None:
        lines.append(f"WARC-Identified-Content-Language: {lang}")
    lines.append("")
    lines += content
    lines += ["", ""]
    return "\n".join(lines) + "\n"


CC_CRAWL = "CC-MAIN-2026-47"
CC_DOC_KEEP1 = [
    "ادبیات فارسی گنجینه‌ای از شعر و نثر کهن است و تاریخ آن به هزاران سال می‌رسد.",
    "شاعران بزرگ این زبان آثار جاودانه‌ای آفریده‌اند که امروز هم خوانده می‌شود.",
    "Home | About | Contact",
    "کتاب‌خانه‌های کهن نیز نقشی بزرگ در نگاه‌داشت این میراث داشته‌اند.",
]
CC_DOC_KEEP2 = [
    "زبان پارسی دری سال‌ها زبان رسمی دربار بود و شاعران آن قصیده‌ها سرودند.",
    "بسیاری از واژه‌های این زبان ریشه در زبان‌های کهن‌تر دارند و پایدار مانده‌اند.",
]
CC_DOC_EVAL = [V_EVAL_SRC + " " + "و متن ادامه می‌یابد تا پنجره بسته شود و شب فرا رسد."]
CC_DOC_KURDISH = ["این متن نامرئی است ڕ و ڵ و ۆ و ێ و ە پایان."]
CC_DOC_URDU = ["یہ متن اردو میں ہے ٹ ڈ ڑ ں ے ختم."]
CC_DOC_TA = ["ةةةةةةةةةة " * 4 + "و اینجا متن فارسی است."]


def cc_wet_gz(slug_docs: list[tuple[str, str | None, list[str]]]) -> bytes:
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as g:
        for i, (lang, uri, content) in enumerate(slug_docs):
            g.write(wet_record("conversion", uri, lang, content).encode("utf-8"))
    return buf.getvalue()


def cc_segments() -> dict[str, bytes]:
    segs: dict[str, bytes] = {}
    segs["seg-01"] = cc_wet_gz([
        ("fas", "http://ex1/a", CC_DOC_KEEP1),
        ("fas", "http://ex1/b", ["کوتاه"]),
        ("ur", "http://ex1/c", CC_DOC_URDU),
        ("fas", "http://ex1/d", CC_DOC_EVAL),
        ("fas", "http://ex1/e", [
            "this document is almost entirely english words so it fails"]),
    ])
    segs["seg-02"] = cc_wet_gz([("fas", "http://ex2/x", CC_DOC_KEEP2)])
    segs["seg-03"] = cc_wet_gz([
        ("fas", "http://ex3/a", CC_DOC_KEEP1),  # exact dup of seg-01/a
        ("fas", "http://ex3/h", CC_DOC_KURDISH),
        ("fa", "http://ex3/i", CC_DOC_TA),
        ("pbu", "http://ex3/j", CC_DOC_KEEP2),  # the pbu quirk: KEPT
        ("fas", "http://ex3/k", CC_DOC_URDU),
        ("fas", "http://ex3/l", ["و" * 100_001]),
    ])
    segs["seg-04"] = cc_wet_gz([("fas", "http://ex4/y", CC_DOC_KEEP1)])
    return segs


CC_PATHS = [f"{CC_CRAWL}/segments/{s}.warc.wet.gz"
            for s in ("seg-01", "seg-02", "seg-03", "seg-04", "seg-05")]


def cc_paths_gz() -> bytes:
    return gzip.compress(("\n".join(CC_PATHS) + "\n").encode())


# ---- eval files ----

def eval_poetry_rows() -> list[dict]:
    return [
        {"src": V_EVAL_SRC, "tgt": vocalize(V_EVAL_SRC)},
    ]


def eval_docs_rows() -> list[dict]:
    return [{"src": "سطری در سند جداگانه",
             "tgt": vocalize("سطری در سند جداگانه")}]


# ---- the local HTTP fixture server ----

class _Handler(BaseHTTPRequestHandler):
    routes: list[tuple[str, bytes, str]] = []  # (path, body, content_type)

    def do_GET(self):  # noqa: N802 (stdlib naming)
        path = urlparse(self.path).path
        for pattern, body, ctype in self.routes:
            if path == pattern:
                self.send_response(200)
                self.send_header("Content-Type", ctype)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
        self.send_response(404)
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_POST(self):  # noqa: N802
        self.do_GET()

    def log_message(self, fmt, *args):  # silence
        pass


class FixtureServer:
    def __init__(self, routes):
        handler = type("H", (_Handler,), {"routes": routes})
        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), handler)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever,
                                       daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()

    @property
    def base(self) -> str:
        return f"http://127.0.0.1:{self.port}"


# ---- reference loader ----

def load_reference(name: str, replacements: dict[str, str] | None = None) -> types.ModuleType:
    """Load a FROZEN pre-refactor script as a module. `replacements`
    rewrites source-base URL literals to the local fixture server — a
    test-harness loading step; the frozen files stay byte-verbatim."""
    path = REFERENCE_DIR / name
    src = path.read_text(encoding="utf-8")
    for old, new in (replacements or {}).items():
        assert old in src, f"{old!r} not found in frozen {name}"
        src = src.replace(old, new)
    mod = types.ModuleType(f"reference_{path.stem}")
    mod.__file__ = str(path)
    exec(compile(src, str(path), "exec"), mod.__dict__)  # noqa: S102
    return mod


# ---- artifact-tree comparators ----

def read_manifest(d: Path) -> dict:
    return json.loads((d / "manifest.json").read_text(encoding="utf-8"))


def norm_manifest(m: dict) -> dict:
    m = copy.deepcopy(m)
    m.pop("built_utc", None)
    return m


def compare_trees(a: Path, b: Path, *, normalize_built_utc: bool = True) -> list[str]:
    """Byte-compare every file in two artifact trees; returns [] on
    equality, else the list of mismatched relative paths (with reasons)."""
    rel_a = sorted(p.relative_to(a) for p in a.rglob("*") if p.is_file())
    rel_b = sorted(p.relative_to(b) for p in b.rglob("*") if p.is_file())
    if rel_a != rel_b:
        return [f"file-set mismatch: {set(rel_a) ^ set(rel_b)}"]
    bad = []
    for rel in rel_a:
        ra, rb = a / rel, b / rel
        if normalize_built_utc and rel.name == "manifest.json":
            if norm_manifest(read_manifest(ra.parent)) != \
                    norm_manifest(read_manifest(rb.parent)):
                bad.append(f"{rel}: manifest differs (beyond built_utc)")
        elif normalize_built_utc and rel.name == "SHA256SUMS.txt":
            la = ra.read_text(encoding="utf-8").splitlines()
            lb = rb.read_text(encoding="utf-8").splitlines()
            if la[:-1] != lb[:-1]:
                bad.append(f"{rel}: shard checksum lines differ")
            if la and lb and la[-1].split("  ")[1:] != lb[-1].split("  ")[1:]:
                bad.append(f"{rel}: manifest entry name differs")
        elif ra.read_bytes() != rb.read_bytes():
            bad.append(f"{rel}: bytes differ")
    return bad


def shard_rows(d: Path, prefix: str) -> list[dict]:
    rows = []
    for shard in sorted(d.glob(f"{prefix}-*.jsonl")):
        for line in shard.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows
