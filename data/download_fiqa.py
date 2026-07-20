from pathlib import Path
from datasets import load_dataset


DATA_DIR = Path(__file__).parent / "fiqa"
DATA_DIR.mkdir(exist_ok=True)

# Pulling 3 pieces of fiqa dataset

corpus = load_dataset("BeIR/fiqa", "corpus", split="corpus")
queries = load_dataset("BeIR/fiqa", "queries", split="queries")
qrels = load_dataset("BeIR/fiqa-qrels", split="test")


# Cache as parquet so other files load instantly

corpus.to_parquet(DATA_DIR / "corpus.parquet")
queries.to_parquet(DATA_DIR / "queries.parquet")
qrels.to_parquet(DATA_DIR / "qrels.parquet")


if __name__ == "__main__":
    print(f"Corpus: {len(corpus):>6} docs")
    print(f"Corpus: {len(queries):>6} queries")
    print(f"Corpus: {len(qrels):>6} judgements")