#!/usr/bin/env python3
import json, asyncio, aiohttp, re, os
from pathlib import Path
H = Path(__file__).resolve().parent
K = os.environ["OPENROUTER_API_KEY"]
SYS = """You label table-QA questions by the reasoning operations they require. Definitions:
- SELECT: pick a row/cell using a key or condition (e.g., "value for the row where Year = 2023"); the key/condition cell is selection evidence, not the answer. A plain lookup of a named row and column is also SELECT.
- COMPARE: compare several candidate values to pick max/min/rank/outlier/which is larger; the answer is the chosen item or its value.
- AGGREGATE: sum/average/count over 3+ cells.
- JOIN: get a key or intermediate value from one table/cell, then use it to find the final value in another table/row (multi-hop).
- MULTI_CELL: answer lists several cell values (set/list) with no arithmetic.
- COMPUTE: arithmetic (difference, ratio, sum) over exactly two operand cells.
- OTHER: none of the above (e.g., describe a trend/pattern in words).
final_op = the LAST operation that produces the answer. pre_steps = earlier operations used to decide which cells are involved, as a list from {SELECT, COMPARE, JOIN} (empty if none).
scope = "SCAN" if the question requires scanning all rows of a column or table to find candidates (max/min/rank/outlier/count over all rows), else "NAMED" (the needed cells are identified by names, keys, or conditions).
table_ref = "CAPTION" if the table is identified by quoting its caption, "NUMBER" if by table number, else "NONE".
Reply with JSON only: {"final_op": ..., "pre_steps": [...], "scope": ..., "table_ref": ..., "note": "<=15 words"}"""
def user(r): return json.dumps({k: r[k] for k in ("question", "gold", "n_tables", "operand_cells", "answer_cell", "key_cells", "n_candidate_cells")}, ensure_ascii=False)
async def one(sess, sem, r):
    body = {"model": "openai/gpt-5", "messages": [{"role": "system", "content": SYS}, {"role": "user", "content": user(r)}], "reasoning": {"effort": "low"}, "max_tokens": 3000}
    err = ""
    async with sem:
        for a in range(4):
            try:
                async with sess.post("https://openrouter.ai/api/v1/chat/completions", headers={"Authorization": f"Bearer {K}"}, json=body, timeout=aiohttp.ClientTimeout(total=240)) as resp:
                    d = await resp.json()
                txt = d["choices"][0]["message"]["content"]; j = json.loads(re.search(r"\{.*\}", txt, re.S).group(0))
                return {"qid": r["qid"], "pool": r["pool"], "gpt5": j, "cost": (d.get("usage") or {}).get("cost")}
            except Exception as e:
                err = str(e); await asyncio.sleep(2 ** a)
    return {"qid": r["qid"], "pool": r["pool"], "error": err}
async def main():
    rows = [json.loads(l) for l in open(H / "items_c0_all.jsonl")]
    out = H / "labels_c0_gpt5.jsonl"; done = set()
    if out.exists(): done = {(json.loads(l)["pool"], json.loads(l)["qid"]) for l in open(out) if "gpt5" in json.loads(l)}
    rows = [r for r in rows if (r["pool"], r["qid"]) not in done]
    sem = asyncio.Semaphore(16)
    async with aiohttp.ClientSession() as s:
        with open(out, "a") as f:
            for fut in asyncio.as_completed([one(s, sem, r) for r in rows]):
                res = await fut; f.write(json.dumps(res, ensure_ascii=False) + "\n"); f.flush()
    print("done", len(rows))
asyncio.run(main())
