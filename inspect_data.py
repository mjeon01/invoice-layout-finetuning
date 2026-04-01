import json

# ── 합성 데이터 ──────────────────────────────────────────
all_s   = json.load(open("data/ground_truth/all_synth.json"))
train_s = json.load(open("data/ground_truth/train_synth.json"))
val_s   = json.load(open("data/ground_truth/val_synth.json"))
tr200   = json.load(open("data/ground_truth/train_synth_200.json"))
tr50    = json.load(open("data/ground_truth/train_synth_50.json"))

print("=" * 55)
print("합성 데이터")
print("=" * 55)
print(f"  all_synth.json       : {len(all_s):>5}건")
print(f"  train_synth.json     : {len(train_s):>5}건")
print(f"  val_synth.json       : {len(val_s):>5}건")
print(f"  train_synth_200.json : {len(tr200):>5}건")
print(f"  train_synth_50.json  : {len(tr50):>5}건")

s0 = all_s[0]
print("\n[첫 번째 합성 샘플]")
print("  meta:", s0["meta"])
print("  y_true 키 & 값:")
for k, v in s0["y_true"].items():
    print(f"    {k:<45} = {str(v)[:60]}")

# 라인 아이템 최대 개수 파악
max_items = 0
for rec in all_s:
    idxs = {int(k.split("[")[1].split("]")[0])
            for k in rec["y_true"] if k.startswith("line_items[")}
    if idxs:
        max_items = max(max_items, max(idxs) + 1)
print(f"\n  합성 데이터 최대 라인 아이템 수: {max_items}")

# ── 실제 데이터 ──────────────────────────────────────────
test_t = json.load(open("data/ground_truth/invoices_test_truth.json"))
val_t  = json.load(open("data/ground_truth/invoices_val_truth.json"))

print()
print("=" * 55)
print("실제 데이터")
print("=" * 55)

# test
if isinstance(test_t[0], list):
    total_test = sum(len(g) for g in test_t)
    print(f"  invoices_test_truth.json : {len(test_t)}그룹 / 총 {total_test}건")
    for i, g in enumerate(test_t):
        print(f"    group[{i}]: {len(g)}건")
    flat_test = [inv for g in test_t for inv in g]
else:
    flat_test = test_t
    print(f"  invoices_test_truth.json : {len(flat_test)}건 (flat)")

if isinstance(val_t[0], list):
    total_val = sum(len(g) for g in val_t)
    print(f"  invoices_val_truth.json  : {len(val_t)}그룹 / 총 {total_val}건")
    for i, g in enumerate(val_t):
        print(f"    group[{i}]: {len(g)}건")
    flat_val = [inv for g in val_t for inv in g]
else:
    flat_val = val_t
    print(f"  invoices_val_truth.json  : {len(flat_val)}건 (flat)")

# 실제 데이터 첫 샘플 구조
real0 = flat_test[0]
print("\n[실제 데이터 첫 샘플]")
print("  keys:", list(real0.keys()))
if "y_true" in real0:
    print("  y_true 키 & 값:")
    for k, v in real0["y_true"].items():
        print(f"    {k:<45} = {str(v)[:60]}")
if "meta" in real0:
    print("  meta:", real0["meta"])
