from contextlib import contextmanager
from pathlib import Path
from . import pdf_loader


class DirScanner:
    def __init__(self, path, skip=()):
        """skip: resolved path strings already processed (e.g. stored in the db)."""
        self.pending = sorted(
            f for f in Path(path).rglob("*.pdf")
            if f.is_file() and str(f.resolve()) not in skip
        )
        self.papers: dict[Path, pdf_loader.Paper] = {}
        self.failed: dict[Path, Exception] = {}

    def __iter__(self):
        return self

    def __len__(self):
        return len(self.pending)

    def __next__(self) -> tuple[Path, pdf_loader.Paper | None]:
        """The next path and its paper, or None if it failed to parse (see self.failed)."""
        if not self.pending:
            raise StopIteration
        path = self.pending.pop(0)
        try:
            paper = pdf_loader.scan_paper(path)
        except Exception as e:  # keep going past broken PDFs
            self.failed[path] = e
            return path, None
        self.papers[path] = paper
        return path, paper