import glob
import sys


def load(paths):
    rows, seen = [], set()
    for p in paths:
        for line in open(p):
            v = line.rstrip("\n").split("\t")
            if len(v) < 11:
                continue
            key = (v[7], v[8], v[9])
            if key in seen:
                continue
            seen.add(key)
            words, dbl, ind, lazy = map(int, v[:4])
            ev = 1.15 * words + (3.3 if ind else 0) + ((3.4 if lazy else -0.7) if ind else 0) + max(0, dbl - 4)
            rows.append(dict(words=words, dbl=dbl, ind=ind, lazy=lazy, key=v[4], rings=v[5], score=int(v[6]),
                             order=v[7], idx=int(v[8]), plugs=v[9], pt=v[10], ev=ev))
    return rows


def show(title, rows, k):
    print(f"\n== {title} ({len(rows)} rows)")
    for r in rows[:k]:
        flag = f"ind {r['key']} {r['rings']}" if r["ind"] else ""
        print(f"{r['ev']:5.1f} w{r['words']:2d} d{r['dbl']} s{r['score']} {r['order']} {r['idx']:8d} {r['pt']}  {flag}{' LAZY' if r['lazy'] else ''}")


def main():
    pattern = sys.argv[1]
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 25
    rows = load(sorted(glob.glob(pattern)))
    show("combined evidence", sorted(rows, key=lambda r: -r["ev"]), k)
    show("word coverage", sorted(rows, key=lambda r: (-r["words"], -r["score"])), k)
    show("indicator fits, by words", sorted([r for r in rows if r["ind"]], key=lambda r: (-r["words"], -r["score"])), k)
    show("indicator fits with lazy key", sorted([r for r in rows if r["lazy"]], key=lambda r: (-r["words"], -r["score"])), k)
    show("doubled substrings >= 6", sorted([r for r in rows if r["dbl"] >= 6], key=lambda r: (-r["dbl"], -r["words"])), k)
    show("search score", sorted(rows, key=lambda r: -r["score"]), k)


if __name__ == "__main__":
    main()
