import argparse
import math
import os
import pathlib
import pypdf
import tqdm
import re
from collections import Counter
from dataclasses import dataclass, field
from pypdf.generic import ByteStringObject
from contextlib import contextmanager

REF_HEADINGS = {"references", "bibliography"}

@dataclass
class Sentence:
    text: str
    page: int
    paragraph: int  # running paragraph index within the paper
    citations: list[int] = field(default_factory=list)  # indices into the paper's citations

@dataclass
class Citation:
    text: str
    page: int
    key: str | None = None  # named destination, e.g. "cite.bert"

@dataclass
class Paper:
    name: str
    title: str | None = None  # from the first page's largest text or the PDF metadata
    sentences: list[Sentence] = field(default_factory=list)
    citations: list[Citation] = field(default_factory=list)

def to_page(cm, tm):
    x, y = tm[4], tm[5]
    return (x * cm[0] + y * cm[2] + cm[4],
            x * cm[1] + y * cm[3] + cm[5])

def page_chunks(page):
    chunks = []
    def visitor(text, cm, tm, font_dict, font_size):
        if text.strip():
            x, y = to_page(cm, tm)
            # the Tf size is often 1 with the scaling done by the text/CTM matrices,
            # so the rendered size is font_size times the combined matrix's scale
            a, b = tm[0] * cm[0] + tm[1] * cm[2], tm[0] * cm[1] + tm[1] * cm[3]
            c, d = tm[2] * cm[0] + tm[3] * cm[2], tm[2] * cm[1] + tm[3] * cm[3]
            chunks.append({
                "x": round(x, 1), "y": round(y, 1), "text": text,
                "font": font_dict.get("/BaseFont") if font_dict else None,
                "size": font_size,
                "render_size": round(font_size * math.sqrt(abs(a * d - b * c)), 1),
                "upright": abs(b) < 1e-3 and a > 0,  # not rotated, e.g. arXiv's side stamp
                "links": set(),
            })

    _skip_textless_forms(page)
    page.extract_text(visitor_text=visitor)
    return chunks

def _skip_textless_forms(page):
    """
    pypdf parses every operator of every Form XObject looking for text, which dominates
    runtime for vector figures (heatmaps, plots). A form with no BT (begin text) operator
    and no nested XObjects can't produce text, so skip it. Hooks a private pypdf method;
    if pypdf changes, extraction just runs unoptimized.
    """
    original = getattr(page, "_extract_text__xform", None)
    if original is None:
        return

    def xform(*, resources_dict, operands, **kwargs):
        try:
            form = resources_dict["/XObject"][operands[0]].get_object()
            if (form.get("/Subtype") == "/Form"
                    and not (form.get("/Resources") or {}).get("/XObject")
                    and b"BT" not in form.get_data()):
                return None
        except Exception:
            pass
        return original(resources_dict=resources_dict, operands=operands, **kwargs)

    page._extract_text__xform = xform

def tag_links(page, chunks, k=3):
    for ref in page.get("/Annots") or []:
        annot = ref.get_object()
        if annot.get("/Subtype") != "/Link":
            continue
        name = dest_name(annot)
        if not name or not name.startswith("cite."):
            continue
        x1, y1, x2, y2 = (float(v) for v in annot["/Rect"])

        def dist(c):  # 0 when the chunk's origin is inside the rect
            dx = max(x1 - c["x"], 0, c["x"] - x2)
            dy = max(y1 - c["y"], 0, c["y"] - y2)
            return math.hypot(dx, dy)

        # the link's text lives in a chunk that starts at or left of the link, so skip
        # chunks starting to its right (e.g. the next column at the same height)
        candidates = [c for c in chunks if c["x"] <= x1 + 1]
        for c in sorted(candidates, key=dist)[:k]:
            c["links"].add(name)

def group_lines(chunks, y_tol=2.0):
    lines = []
    for c in sorted(chunks, key=lambda c: (-c["y"], c["x"])):
        if lines and abs(lines[-1][0]["y"] - c["y"]) <= y_tol:
            lines[-1].append(c)
        else:
            lines.append([c])
    return [sorted(line, key=lambda c: c["x"]) for line in lines]

def column_starts(chunks, min_count=3, gap=30.0):
    counts = Counter(c["x"] for c in chunks if any(ch.isalnum() for ch in c["text"]))
    xs = sorted(x for x, n in counts.items() if n >= min_count)
    clusters = []
    for x in xs:
        if clusters and x - clusters[-1][-1] <= gap:
            clusters[-1].append(x)
        else:
            clusters.append([x])
    return [cluster[0] for cluster in clusters]

def column_lines(chunks, y_tol=2.0, min_count=3, gap=30.0, margin_tol=1.0):
    if not chunks:
        return []
    starts = column_starts(chunks, min_count, gap) or [min(c["x"] for c in chunks)]
    columns = {}
    for c in chunks:
        i = max((k for k, s in enumerate(starts) if s <= c["x"] + margin_tol), default=0)
        columns.setdefault(i, []).append(c)
    return [(starts[i], line) for i in sorted(columns) for line in group_lines(columns[i], y_tol)]

def _line_text(line):
    text = re.sub(r"\s+", " ", "".join(c["text"] for c in line).replace("\n", " ")).strip()
    return text

def _join(text, more):
    if not text:
        return more
    return text[:-1] + more if text.endswith("-") else text + " " + more

def group_citations(lines, margin_tol=1.0):
    citations = []
    current, current_margin = None, None
    for margin, line in lines:
        text = _line_text(line)
        if not text:
            continue
        if current is None or margin != current_margin or line[0]["x"] <= margin + margin_tol:
            if current is not None:
                citations.append(current)
            # lines kept so non-bibliography entries can be handed back to the body
            current = {"text": text, "x": line[0]["x"], "y": line[0]["y"], "lines": []}
            current_margin = margin
        else:
            current["text"] = _join(current["text"], text)
        current["lines"].append((margin, line))
    if current is not None:
        citations.append(current)
    return citations

def match_destination(citations, page, left, top, tol=2.0):
    best, best_dist = None, None
    for cite in citations:
        if cite["page"] != page:
            continue
        dx, dy = cite["x"] - left, top - cite["y"]
        if dx < -tol or dy < -tol:
            continue
        dist = dx + dy
        if best_dist is None or dist < best_dist:
            best, best_dist = cite, dist
    return best

def dest_name(annot):
    dest = annot.get("/Dest")
    action = annot.get('/A')
    if action and not dest:
        action = action.get_object()
        if action.get("/S") == "/GoTo":
            dest = action.get("/D")
    if isinstance(dest, ByteStringObject):
        dest = dest.decode("latin-1")
    return dest if isinstance(dest, str) else None

# tokens before a "." that don't end a sentence
ABBREVIATIONS = {"al", "e.g", "i.e", "etc", "cf", "vs", "fig", "figs", "sec", "eq", "eqs", "no", "vol", "resp"}

def group_paragraphs(body, gap_factor=1.4, indent_tol=3.0, small_font=0.75):
    gaps = sorted(
        a[2][0]["y"] - b[2][0]["y"]
        for a, b in zip(body, body[1:])
        if a[:2] == b[:2] and a[2][0]["y"] > b[2][0]["y"]
    )
    spacing = gaps[len(gaps) // 2] if gaps else None
    sizes = sorted(c["size"] for _, _, line in body for c in line)
    body_size = sizes[len(sizes) // 2] if sizes else 0

    paragraphs, prev = [], None
    for pno, margin, line in body:
        x, y = line[0]["x"], line[0]["y"]
        small = all(c["size"] < small_font * body_size for c in line)
        if prev is None:
            new = True
        elif (pno, margin) == prev[:2]:
            gap = prev[3] - y
            if small or (spacing and abs(gap) < 0.5 * spacing):
                # raised/lowered bits of the same line (superscripts, footnote marks):
                # keep them in the paragraph without moving the reference line
                paragraphs[-1].append((pno, line))
                continue
            new = (gap < 0 or (spacing and gap > gap_factor * spacing)
                   or x > prev[2] + indent_tol)
        else:  # top of a new column or page: continue unless the line is indented,
            # or pages were skipped in between (e.g. the bibliography)
            new = x > margin + indent_tol or pno > prev[0] + 1
        if new:
            paragraphs.append([])
        paragraphs[-1].append((pno, line))
        prev = (pno, margin, x, y)
    return paragraphs

ABBREVIATIONS = {"al", "e.g", "i.e", "etc", "cf", "vs", "fig", "figs", "sec", "eq", "eqs", "no", "vol", "resp"}

def _sentence_ends(text):
    for m in re.finditer(r"([.!?][\"”’)\]]*|:)\s+(?=\S)", text):
        if m.group(1) != ":":
            if text[m.end()].islower():
                continue  # "... 2 vs. some", mid-sentence period
            prev = re.search(r"([\w.]+)$", text[:m.start()])
            token = prev.group(1).lower() if prev else ""
            if token in ABBREVIATIONS or (len(token) == 1 and token.isalpha()):
                continue  # "et al.", "e.g.", initials like "D."
        yield m.end()

def split_sentences(body, key_index):
    result = []
    for pid, paragraph in enumerate(group_paragraphs(body)):
        # flatten to a stream of characters, each carrying (page, cite keys of its chunk)
        chars, tags = [], []
        for pno, line in paragraph:
            while chars and chars[-1] == " ":
                chars.pop(); tags.pop()
            if chars and chars[-1] == "-":  # dehyphenate across lines
                chars.pop(); tags.pop()
            elif chars:
                chars.append(" "); tags.append((pno, frozenset()))
            for c in line:
                keys = frozenset(c["links"])
                for ch in c["text"].replace("\n", " "):
                    if ch.isspace() and (not chars or chars[-1] == " "):
                        continue
                    chars.append(" " if ch.isspace() else ch)
                    tags.append((pno, keys))
        text = "".join(chars)

        bounds = [0, *_sentence_ends(text), len(text)]
        for s, e in zip(bounds, bounds[1:]):
            sentence = text[s:e].strip()
            if not sentence:
                continue
            keys = set().union(*(k for _, k in tags[s:e]))
            cites = sorted({key_index[k] for k in keys if k in key_index})
            result.append(Sentence(text=sentence, page=tags[s][0], paragraph=pid, citations=cites))
    return result

def _words(text):
    return re.findall(r"[^\W\d_]+", text.lower())

def _clean(text):
    return re.sub(r"\s+", " ", text).strip()

def _mostly_letters(text, share=0.6):
    """Rules out ids, numbers and garbled font encodings (non-ASCII junk)."""
    chars = text.replace(" ", "")
    return sum(ch.isascii() and ch.isalpha() for ch in chars) >= share * len(chars)

def find_title(chunks, meta_title=None, min_words=2, max_words=40, size_tol=0.3):
    """
    Title of a paper from its first page's text chunks. Prefers the PDF metadata title
    when every one of its words appears on the page (metadata is often a file name or
    a typesetting id); otherwise the largest upright text that is bigger than the body
    text, has at least min_words words and is mostly letters (rules out drop caps,
    rotated margin stamps and garbled font encodings). None if nothing qualifies.
    """
    page_words = set(_words(" ".join(c["text"] for c in chunks)))
    meta = _clean(meta_title or "")
    meta_words = _words(meta)
    if (len(meta_words) >= min_words and page_words.issuperset(meta_words)
            and _mostly_letters(meta)  # not an id like "doi:10.1016/…" or "PII: S0010-…"
            and not ("_" in meta and " " not in meta)):  # nor a file name like "BERT_Guide"
        return meta

    upright = [c for c in chunks if c["upright"] and _words(c["text"])]
    if not upright:
        return None
    letters = Counter()
    for c in upright:
        letters[c["render_size"]] += len(c["text"])
    body_size = letters.most_common(1)[0][0]
    sizes = {c["render_size"] for c in upright if c["render_size"] > body_size + size_tol}
    for size in sorted(sizes, reverse=True):
        text = _clean(" ".join(c["text"] for c in upright if abs(c["render_size"] - size) <= size_tol))
        if min_words <= len(_words(text)) <= max_words and _mostly_letters(text):
            return text
    return None

def scan_paper(path) -> Paper:
    """Parse one PDF into its bibliography citations and body sentences."""
    fp = pathlib.Path(path)
    if not fp.is_file():
        raise FileNotFoundError(f"{path} is not a PDF file")
    reader = pypdf.PdfReader(fp)
    named = (reader.named_destinations)
    
    cite_dests = {n: d for n, d in named.items() if n.startswith("cite.")}
    bib_pages = {reader.get_destination_page_number(d) for d in cite_dests.values()}

    seen_heading = False
    paper_citations = []
    
    body_by_page = {}  # page -> [(column margin, line)] of non-bibliography text
    title = None
    pages = tqdm.tqdm(reader.pages, desc=fp.name[:30], unit="page", leave=False, position=1)
    for pno, page in enumerate(pages):
        chunks = page_chunks(page)
        if pno == 0:
            meta = reader.metadata
            title = find_title(chunks, meta.title if meta else None)
        tag_links(page, chunks)
        lines = column_lines(chunks)
        is_bib = pno in bib_pages if bib_pages else seen_heading
        body_lines, ref_lines = ([], lines) if is_bib else (lines, [])
        if not seen_heading and (is_bib or not bib_pages):
            for i, (_, line) in enumerate(lines):
                if _line_text(line).lower() in REF_HEADINGS:
                    seen_heading = True
                    body_lines, ref_lines = lines[:i], lines[i + 1:]
                    break
        body_by_page[pno] = body_lines
        for cite in group_citations(ref_lines):
            cite["page"] = pno
            paper_citations.append(cite)

    # cite key (e.g. "cite.bert") -> entry in paper_citations
    by_key = {}
    for name, dest in cite_dests.items():
        match = match_destination(
            paper_citations, reader.get_destination_page_number(dest),
            float(dest.left or 0), float(dest.top or 0),
        )
        if match:
            by_key[name] = match

    # entries before the first / after the last destination-matched one are
    # non-bibliography text sharing a page with it (e.g. an Appendix after the
    # references): hand their lines back to the body
    if by_key:
        matched = {id(c) for c in by_key.values()}
        hits = [i for i, c in enumerate(paper_citations) if id(c) in matched]
        first, last = hits[0], hits[-1]
        for c in reversed(paper_citations[:first]):  # reversed: each is prepended
            body_by_page[c["page"]][:0] = c["lines"]
        for c in paper_citations[last + 1:]:
            body_by_page[c["page"]].extend(c["lines"])
        paper_citations = paper_citations[first:last + 1]
    body = [(pno, margin, line) for pno in sorted(body_by_page) for margin, line in body_by_page[pno]]

    index_of = {id(c): i for i, c in enumerate(paper_citations)}
    key_index = {k: index_of[id(c)] for k, c in by_key.items()}
    key_of = {i: k for k, i in key_index.items()}
    paper = Paper(
        name=fp.name,
        title=title,
        sentences=split_sentences(body, key_index),
        citations=[Citation(text=c["text"], page=c["page"], key=key_of.get(i))
                   for i, c in enumerate(paper_citations)],
    )
    return paper

def write_sentences(paper, out_path="file.txt"):
    """One sentence per line, a blank line between paragraphs, then the indexed citations."""
    with open(out_path, "w") as file:
        file.write(f"===== {paper.name}\n")
        last = None
        for s in paper.sentences:
            if last is not None and s.paragraph != last:
                file.write("\n")
            last = s.paragraph
            cites = f"  -> {s.citations}" if s.citations else ""
            file.write(f"[p{s.page} ¶{s.paragraph}] {s.text}{cites}\n")
        file.write("\n----- citations\n")
        for i, cite in enumerate(paper.citations):
            file.write(f"[{i}] {cite.text}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path", help="PDF file to load")
    args = parser.parse_args()
    write_sentences(scan_paper(args.path))
