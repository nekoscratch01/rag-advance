import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas.core.config import get_settings  # noqa: E402
from atlas.datasets.financebench_importer import import_prepared_financebench  # noqa: E402
from atlas.db.session import SessionLocal, init_db  # noqa: E402
from atlas.embeddings.bge_local import LocalBGEEmbedder  # noqa: E402
from atlas.embeddings.bm25_sparse import BM25SparseEncoder  # noqa: E402
from atlas.vector.qdrant_client import get_qdrant_client  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import prepared FinanceBench parent/child artifacts into Atlas V1 storage."
    )
    parser.add_argument("--corpus", default="corpus/financebench")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete existing FinanceBench rows/vectors before import.",
    )
    parser.add_argument(
        "--allow-dense-only",
        action="store_true",
        help="Allow import without BM25 sparse vectors. Not recommended for V1 benchmark.",
    )
    args = parser.parse_args()

    settings = get_settings()
    init_db()
    with SessionLocal() as db:
        result = import_prepared_financebench(
            db,
            corpus_dir=args.corpus,
            settings=settings,
            embedder=LocalBGEEmbedder(settings),
            qdrant=get_qdrant_client(),
            sparse_encoder=BM25SparseEncoder(settings),
            batch_size=args.batch_size,
            reset_existing=not args.no_reset,
            require_hybrid=not args.allow_dense_only,
        )

    print(
        "Imported FinanceBench: "
        f"{result.document_count} documents, "
        f"{result.parent_count} parents, "
        f"{result.child_count} children, "
        f"{result.vector_count} vectors."
    )
    print(f"Collection: {result.collection}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
