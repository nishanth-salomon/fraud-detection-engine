"""Command line: python -m fraud_engine {demo,generate,score}"""
from __future__ import annotations

import argparse
import json
import sys

from . import pipeline
from .synth import generate


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="fraud_engine", description=__doc__)
    sub = p.add_subparsers(dest="cmd")

    d = sub.add_parser("demo", help="generate data, train, evaluate, write artifacts/")
    d.add_argument("--customers", type=int, default=600)
    d.add_argument("--days", type=int, default=90)
    d.add_argument("--seed", type=int, default=7)
    d.add_argument("--out", default="artifacts")
    d.add_argument("--review-cost", type=float, default=pipeline.REVIEW_COST)

    g = sub.add_parser("generate", help="write a synthetic transaction CSV")
    g.add_argument("--customers", type=int, default=600)
    g.add_argument("--days", type=int, default=90)
    g.add_argument("--seed", type=int, default=7)
    g.add_argument("--out", default="data/transactions.csv")

    s = sub.add_parser("score", help="score a transaction CSV with a trained model")
    s.add_argument("--model", default="artifacts/model.joblib")
    s.add_argument("--input", required=True)
    s.add_argument("--output", default="artifacts/scored.csv")
    s.add_argument("--threshold", type=float, default=None)

    a = p.parse_args(argv)
    if a.cmd in (None, "demo"):
        args = a if a.cmd else p.parse_args(["demo"])
        res = pipeline.run(args.customers, args.days, args.seed, args.out, args.review_cost)
        t = res["test_at_threshold"]
        print(f"\ncaught {t['fraud_caught']}/{t['fraud_total']} fraud "
              f"(precision {t['precision']:.1%}), net saving Rs {t['saved']:,.0f}")
        print(f"report: {args.out}/report.md")
    elif a.cmd == "generate":
        from pathlib import Path
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        df = generate(a.customers, a.days, a.seed)
        df.to_csv(a.out, index=False)
        print(f"wrote {len(df):,} rows ({df.is_fraud.mean():.2%} fraud) to {a.out}")
    elif a.cmd == "score":
        res = pipeline.score_csv(a.model, a.input, a.output, a.threshold)
        print(f"scored {len(res):,} rows -> {a.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
