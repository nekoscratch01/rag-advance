import argparse
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from atlas.datasets.financebench import prepare_financebench  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare the FinanceBench corpus and eval cases.")
    parser.add_argument("--out", default="corpus/financebench", help="Output corpus directory.")
    parser.add_argument(
        "--evals",
        default="evals/financebench_cases.yaml",
        help="Output eval cases YAML path.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Limit FinanceBench rows.")
    parser.add_argument("--revision", default=None, help="Hugging Face dataset revision.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail instead of recording conflicts.",
    )
    args = parser.parse_args()

    result = prepare_financebench(
        out_dir=args.out,
        evals_path=args.evals,
        limit=args.limit,
        strict=args.strict,
        revision=args.revision,
    )
    print(
        "Prepared FinanceBench: "
        f"{result.row_count} rows, "
        f"{result.manifest_count} manifest records, "
        f"{result.page_count} pages, "
        f"{result.chunk_count} chunks, "
        f"{result.failure_count} failures."
    )
    print(f"Corpus: {result.out_dir}")
    print(f"Evals: {result.evals_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
