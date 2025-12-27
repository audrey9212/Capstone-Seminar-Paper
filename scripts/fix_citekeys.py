#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

CITEKEY_RE = re.compile(r"@([A-Za-z0-9_:\-]+)")
YEAR_RE = re.compile(r"(19\d{2}|20\d{2})")

def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())

def split_bib_entries(bib_text: str) -> List[Tuple[str, str]]:
    """
    Return list of (key, raw_entry_text)
    """
    parts = re.split(r"(?=@\w+\{)", bib_text)
    out = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        m = re.match(r"@\w+\{([^,]+),", p, flags=re.S)
        if not m:
            continue
        key = m.group(1).strip()
        out.append((key, p))
    return out

def extract_citekeys(text: str) -> List[str]:
    return list(set(CITEKEY_RE.findall(text)))

def author_tokens_from_missing_key(missing_key: str) -> Tuple[List[str], str | None]:
    m = YEAR_RE.search(missing_key)
    if not m:
        return [], None
    year = m.group(1)
    prefix = missing_key[:m.start()]
    # split camelCase-ish into tokens
    parts = re.findall(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])", prefix)
    toks = [norm(x) for x in parts if norm(x)]
    toks = [t for t in toks if t not in {"et", "al", "etal"}]
    return toks, year

def propose_replacement(missing_key: str, bib_keys: List[str], bib_entries: List[Tuple[str,str]]) -> Tuple[str | None, List[str], str]:
    """
    Strategy:
      1) Prefix match on bib key: bib_key startswith missing_key (case-insensitive normalized)
      2) Token+year match in bib key string
      3) Token+year match in bib entry body (author/title etc.)
    """
    mk = norm(missing_key)

    # (1) prefix match
    pref = [k for k in bib_keys if norm(k).startswith(mk)]
    if len(pref) == 1:
        return pref[0], pref, "prefix-unique"
    if len(pref) > 1:
        return None, pref[:10], "prefix-ambiguous"

    toks, year = author_tokens_from_missing_key(missing_key)
    if not year or not toks:
        return None, [], "no-year-or-tokens"

    # (2) match in key string
    c2 = []
    for k in bib_keys:
        nk = norm(k)
        if year in nk and all(t in nk for t in toks):
            c2.append(k)
    if len(c2) == 1:
        return c2[0], c2, "key-token-year-unique"
    if len(c2) > 1:
        return None, c2[:10], "key-token-year-ambiguous"

    # (3) match in entry body
    c3 = []
    for k, raw in bib_entries:
        rr = norm(raw)
        if year in rr and all(t in rr for t in toks):
            c3.append(k)
    if len(c3) == 1:
        return c3[0], c3, "entry-token-year-unique"
    if len(c3) > 1:
        return None, c3[:10], "entry-token-year-ambiguous"

    return None, [], "no-candidate"

def safe_replace(text: str, old: str, new: str) -> Tuple[str, int]:
    # replace @old when followed by a non-key char (prevents partial match)
    pat = re.compile(r"@" + re.escape(old) + r"(?=[^A-Za-z0-9_:\-])")
    return pat.subn("@" + new, text)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bib", default="report/references.bib")
    ap.add_argument("--qmd-root", default="report")
    ap.add_argument("--include-root-index", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--apply", action="store_true")
    args = ap.parse_args()

    if args.dry_run == args.apply:
        raise SystemExit("Choose exactly one: --dry-run or --apply")

    bib_path = Path(args.bib)
    qmd_root = Path(args.qmd_root)

    bib_text = bib_path.read_text(encoding="utf-8", errors="ignore")
    bib_entries = split_bib_entries(bib_text)
    bib_keys = [k for k, _ in bib_entries]
    bib_key_set = set(bib_keys)

    qmd_files = sorted(qmd_root.rglob("*.qmd"))
    if args.include_root_index and Path("index.qmd").exists():
        qmd_files = [Path("index.qmd")] + qmd_files

    used = {}
    for f in qmd_files:
        t = f.read_text(encoding="utf-8", errors="ignore")
        for k in extract_citekeys(t):
            used.setdefault(k, set()).add(f)

    missing = sorted([k for k in used if k not in bib_key_set])

    print(f"Bib entries: {len(bib_keys)}")
    print(f"QMD files scanned: {len(qmd_files)}")
    print(f"Unique citekeys used: {len(used)}")
    print(f"Missing citekeys: {len(missing)}\n")

    repl: Dict[str, str] = {}
    unresolved = []

    for k in missing:
        chosen, cands, reason = propose_replacement(k, bib_keys, bib_entries)
        files_preview = ", ".join(sorted({p.name for p in used[k]}))
        print(f"- {k}  (files: {files_preview})")
        if chosen:
            repl[k] = chosen
            print(f"  -> replace with: {chosen}  [{reason}]")
        else:
            unresolved.append(k)
            if cands:
                print(f"  candidates: {cands}  [{reason}]")
            else:
                print(f"  candidates: (none)  [{reason}]")

    print(f"\nProposed auto-replacements: {len(repl)}")
    print(f"Unresolved: {len(unresolved)}")

    if args.dry_run:
        print("\nDRY RUN complete. No files modified.")
        return

    total = 0
    for f in qmd_files:
        text = f.read_text(encoding="utf-8", errors="ignore")
        changed = 0
        for old, new in repl.items():
            text, n = safe_replace(text, old, new)
            changed += n
        if changed:
            f.write_text(text, encoding="utf-8")
            total += changed
            print(f"Applied {changed} replacements in {f}")

    print(f"\nAPPLY complete. Total replacements: {total}")
    if unresolved:
        print("\nStill unresolved (likely not in your bib or ambiguous):")
        for k in unresolved:
            print(f"  - {k}")

if __name__ == "__main__":
    main()
