import json
import re
import sqlite3
from itertools import groupby

import bm25s
import numpy as np
import sqlite_vec
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from . import config
from .dir_scanner import DirScanner


MAX_TOKENS = 512
MIN_TOKENS = 4
EMBED_DIM = 128
MIN_TITLE_WORDS = 5  # titles need more than 4 word tokens to be linked

MODEL_NAME, _ = config.get_model()
model = SentenceTransformer(MODEL_NAME, device="cpu", truncate_dim=EMBED_DIM)

SCHEMA = f"""
CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS files (
    id    INTEGER PRIMARY KEY,
    path  TEXT UNIQUE NOT NULL,
    title TEXT                      -- first paragraph's text
);
CREATE TABLE IF NOT EXISTS citations (
    id      INTEGER PRIMARY KEY,
    file_id INTEGER NOT NULL REFERENCES files(id),
    link_id INTEGER REFERENCES files(id),  -- the indexed file this citation refers to
    idx     INTEGER NOT NULL,       -- position in the paper's bibliography
    key     TEXT,                   -- named destination, e.g. cite.bert
    text    TEXT NOT NULL,
    UNIQUE (file_id, idx),
    UNIQUE (file_id, text)
);
CREATE TABLE IF NOT EXISTS chunks (
    id        INTEGER PRIMARY KEY,  -- same as vec_chunks.rowid
    file_id   INTEGER NOT NULL REFERENCES files(id),
    paragraph INTEGER NOT NULL,
    page      INTEGER NOT NULL,
    tokens    INTEGER NOT NULL,
    text      TEXT NOT NULL,
    citations TEXT NOT NULL         -- JSON list of citations.id
);
CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(
    embedding float[{EMBED_DIM}] distance_metric=cosine
);
"""

def connect(db_path):
    con = sqlite3.connect(db_path)
    con.enable_load_extension(True)
    sqlite_vec.load(con)
    con.enable_load_extension(False)
    con.executescript(SCHEMA)
    check_model(con, db_path)
    return con

def check_model(con, db_path):
    """A db's vectors only make sense to the model that made them: record it, refuse others."""
    dim = model.get_embedding_dimension()
    if dim != EMBED_DIM:
        raise SystemExit(f"model {MODEL_NAME} gives {dim}-dim embeddings; the index needs {EMBED_DIM}")
    row = con.execute("SELECT value FROM meta WHERE key = 'model'").fetchone()
    if row is None:
        with con:
            con.execute("INSERT INTO meta (key, value) VALUES ('model', ?)", (MODEL_NAME,))
    elif row[0] != MODEL_NAME:
        raise SystemExit(
            f"{db_path} was built with {row[0]}, but the current model is {MODEL_NAME}.\n"
            f"Switch back with `arximedes model {row[0]}`, or use a different --db."
        )

def count_tokens(texts):
    """Content tokens per text, without [CLS]/[SEP] (added once per chunk instead)."""
    return [len(ids) for ids in token_ids(texts, special=False)]

def token_ids(texts, special):
    tok = model.tokenizer
    if hasattr(tok, "encode_batch"):  # tokenizers.Tokenizer, used by static embedding models
        return [enc.ids for enc in tok.encode_batch(texts, add_special_tokens=special)]
    return tok(texts, add_special_tokens=special)["input_ids"]  # transformers tokenizer

SPECIAL_TOKENS = len(token_ids([""], special=True)[0])  # [CLS] + [SEP] the model adds per input

def chunk_paper(paper, max_tokens=MAX_TOKENS, min_tokens=MIN_TOKENS):
    """
    One chunk per paragraph. If adding a sentence would push the chunk past max_tokens,
    the chunk ends before it and that sentence starts the next one. A single sentence
    over the limit becomes a chunk of its own. Chunks under min_tokens (stray page
    numbers, figure labels) are dropped.
    """
    tokens = count_tokens([s.text for s in paper.sentences])
    sized = list(zip(paper.sentences, tokens))
    chunks = []
    for _, group in groupby(sized, key=lambda st: st[0].paragraph):
        current, current_tokens = [], SPECIAL_TOKENS
        for sentence, n in group:
            if current and current_tokens + n > max_tokens:
                chunks.append((current, current_tokens))
                current, current_tokens = [], SPECIAL_TOKENS
            current.append(sentence)
            current_tokens += n
        if current:
            chunks.append((current, current_tokens))
    return [(sentences, n) for sentences, n in chunks if n >= min_tokens]

def store_paper(con, path, paper):
    chunks = chunk_paper(paper)
    texts = [" ".join(s.text for s in sentences) for sentences, _ in chunks]
    embeddings = model.encode(texts, normalize_embeddings=True).astype(np.float32)

    title = paper.title
    if not title:
        first_para = paper.sentences[0].paragraph if paper.sentences else None
        title = " ".join(s.text for s in paper.sentences if s.paragraph == first_para)

    with con:
        file_id = con.execute(
            "INSERT INTO files (path, title) VALUES (?, ?)", (path, title)
        ).lastrowid

        cite_ids = {}
        for i, c in enumerate(paper.citations):
            if not c.key:
                continue
            con.execute(
                "INSERT OR IGNORE INTO citations (file_id, idx, key, text) VALUES (?, ?, ?, ?)",
                (file_id, i, c.key, c.text),
            )
            cite_ids[i] = con.execute(
                "SELECT id FROM citations WHERE file_id = ? AND text = ?", (file_id, c.text)
            ).fetchone()[0]
        for (sentences, n_tokens), text, emb in zip(chunks, texts, embeddings):
            cites = sorted({cite_ids[i] for s in sentences for i in s.citations if i in cite_ids})
            chunk_id = con.execute(
                "INSERT INTO chunks (file_id, paragraph, page, tokens, text, citations) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (file_id, sentences[0].paragraph, sentences[0].page, n_tokens, text, json.dumps(cites)),
            ).lastrowid
            con.execute(
                "INSERT INTO vec_chunks (rowid, embedding) VALUES (?, ?)", (chunk_id, emb.tobytes())
            )
    return file_id, len(chunks)

def word_tokens(text):
    """Lowercased alphabetic tokens; drops numbers and symbols like affiliation marks."""
    return [w.lower() for w in re.findall(r"[^\W\d_]+", text)]

def link_citations(con, k=50):
    """
    Point each citation at the indexed file it refers to. Every file title with at
    least MIN_TITLE_WORDS word tokens is run as a BM25 query over the citation texts;
    a hit counts only if all of the title's word tokens appear in the citation, and a
    citation matched by several files keeps the best-scoring one. Recomputes all links.
    """
    rows = con.execute("SELECT id, file_id, text FROM citations").fetchall()
    con.execute("UPDATE citations SET link_id = NULL")
    if not rows:
        con.commit()
        return 0
    cite_ids, owners, texts = zip(*rows)
    cite_words = [set(word_tokens(t)) for t in texts]
    retriever = bm25s.BM25()
    retriever.index(bm25s.tokenize(list(texts), stopwords=None, show_progress=False), show_progress=False)

    files = [(fid, word_tokens(title or "")) for fid, title in con.execute("SELECT id, title FROM files")]
    files = [(fid, words) for fid, words in files if len(words) >= MIN_TITLE_WORDS]
    if not files:
        con.commit()
        return 0
    docs, scores = retriever.retrieve(
        [words for _, words in files], k=min(k, len(texts)), show_progress=False
    )

    best = {}  # citation row index -> (score, file id)
    for (fid, words), row_docs, row_scores in zip(files, docs, scores):
        for d, score in zip(row_docs, row_scores):
            if score <= 0 or owners[d] == fid or not cite_words[d].issuperset(words):
                continue
            if d not in best or score > best[d][0]:
                best[d] = (score, fid)
    with con:
        con.executemany(
            "UPDATE citations SET link_id = ? WHERE id = ?",
            [(fid, cite_ids[d]) for d, (_, fid) in best.items()],
        )
    return len(best)

def index_dir(con, path):
    """Parse, chunk, embed and store every PDF under path that isn't in the db yet."""
    known = {row[0] for row in con.execute("SELECT path FROM files")}
    scanner = DirScanner(path, skip=known)
    stored = cites = 0
    for pdf, paper in tqdm(scanner, total=len(scanner), desc="papers", unit="paper", position=0):
        if paper is None:
            continue
        file_id, _ = store_paper(con, str(pdf.resolve()), paper)
        stored += 1
        cites += con.execute("SELECT count(*) FROM citations WHERE file_id = ?", (file_id,)).fetchone()[0]
    linked = link_citations(con)
    total_files, total_cites = con.execute(
        "SELECT (SELECT count(*) FROM files), (SELECT count(*) FROM citations)"
    ).fetchone()
    failed = f", {len(scanner.failed)} failed" if scanner.failed else ""
    print(f"indexed {stored} papers with {cites} citations{failed}; "
          f"db has {total_files} papers, {total_cites} citations, {linked} linked to a paper")
