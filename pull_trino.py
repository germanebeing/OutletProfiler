"""Minimal read-only Trino REST client for exploration."""
import os
import sys, time, json, httpx

BASE = os.environ.get("TRINO_URL", "http://localhost:8080")
H = {"X-Trino-User": "admin", "X-Presto-User": "admin"}


def run(sql, catalog=None, schema=None, timeout=60):
    h = dict(H)
    if catalog:
        h["X-Trino-Catalog"] = catalog
    if schema:
        h["X-Trino-Schema"] = schema
    r = httpx.post(BASE + "/v1/statement", content=sql.encode(), headers=h, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    cols, rows = [], []
    guard = 0
    while True:
        guard += 1
        if guard > 2000:
            break
        if j.get("columns") and not cols:
            cols = [c["name"] for c in j["columns"]]
        if j.get("data"):
            rows.extend(j["data"])
        if j.get("error"):
            raise RuntimeError(json.dumps(j["error"].get("message", j["error"])))
        nu = j.get("nextUri")
        if not nu:
            break
        time.sleep(0.08)
        rr = httpx.get(nu, headers=H, timeout=timeout)
        rr.raise_for_status()
        j = rr.json()
    return cols, rows


def show(sql, catalog=None, schema=None, maxrows=60):
    cols, rows = run(sql, catalog, schema)
    print(f"# {sql}")
    if cols:
        print(" | ".join(cols))
    for row in rows[:maxrows]:
        print(" | ".join("" if v is None else str(v) for v in row))
    if len(rows) > maxrows:
        print(f"... (+{len(rows)-maxrows} more rows, {len(rows)} total)")
    print()


if __name__ == "__main__":
    sql = sys.argv[1]
    cat = sys.argv[2] if len(sys.argv) > 2 else None
    sch = sys.argv[3] if len(sys.argv) > 3 else None
    show(sql, cat, sch)
