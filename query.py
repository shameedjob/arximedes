from pathlib import Path

import numpy as np

from .embedder import model
from .pdf_loader import scan_paper


def search(con, query, k=10, y=5, span=3):
    """
    Two-stage retrieval. The k chunks nearest the query pick the candidate documents;
    those PDFs are re-scanned, every run of `span` consecutive sentences is embedded and
    scored against the query, and the best non-overlapping spans are kept.

    Returns the y best spans as (text, path, page, score), best first, where page is the
    0-based page of the span's first sentence and score is cosine similarity.
    """
    q = model.encode(query, normalize_embeddings=True).astype(np.float32)

    paths = [row[0] for row in con.execute(
        """
        WITH knn AS (
            SELECT rowid, distance FROM vec_chunks WHERE embedding MATCH ? AND k = ?
        )
        SELECT f.path FROM knn JOIN chunks c ON c.id = knn.rowid JOIN files f ON f.id = c.file_id
        GROUP BY f.id ORDER BY min(knn.distance)
        """,
        (q.tobytes(), k),
    )]

    spans = []  # (path, start, end, text, page): sentences [start, end) of one document
    for path in paths:
        if not Path(path).exists():  # moved or deleted since it was indexed
            continue
        sents = scan_paper(path).sentences
        for i in range(max(1, len(sents) - span + 1)):  # a short paper is one span
            window = sents[i:i + span]
            if window:
                spans.append((path, i, i + len(window), " ".join(s.text for s in window), window[0].page))
    if not spans:
        return []

    scores = model.encode([s[3] for s in spans], normalize_embeddings=True) @ q
    results, taken = [], []  # taken: (path, start, end) of spans already reported
    for i in np.argsort(-scores):
        path, start, end, text, page = spans[i]
        if any(p == path and start < e and s < end for p, s, e in taken):
            continue  # shares sentences with a better span
        taken.append((path, start, end))
        results.append((text, path, page, float(scores[i])))
        if len(results) == y:
            break
    return results


# def get_context(chunk_id):
#     """sumary_line
    
#     Keyword arguments:
#     argument -- description
#     Return: return_description
#     """
    
#     pass

# def cites(paper):
#     pass

# def cited_by(paper):
#     pass

# def citation_contexts(ref):
#     pass

# def locate(chunk_id):
#     pass

# def retrieve_from_paper(paper, query):
#     pass