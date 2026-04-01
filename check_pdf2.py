import fitz, json

# ── 합성 PDF 첫 페이지 텍스트 추출 테스트 ──
doc = fitz.open("data/input/synth/all_synth.pdf")
page = doc[0]
blocks = page.get_text("words")  # (x0, y0, x1, y1, word, block_no, line_no, word_no)
print("=== 합성 PDF 1페이지 텍스트 (첫 20 토큰) ===")
for b in blocks[:20]:
    x0,y0,x1,y1,word = b[0],b[1],b[2],b[3],b[4]
    print(f"  '{word}'  bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")
print(f"  총 토큰 수: {len(blocks)}")
print(f"  페이지 크기: {page.rect.width:.0f} x {page.rect.height:.0f} pt")
doc.close()

print()

# ── 실제 PDF 첫 페이지 ──
doc2 = fitz.open("data/input/invoices_val.pdf")
page2 = doc2[0]
blocks2 = page2.get_text("words")
print("=== 실제 PDF (val) 1페이지 텍스트 (첫 20 토큰) ===")
for b in blocks2[:20]:
    x0,y0,x1,y1,word = b[0],b[1],b[2],b[3],b[4]
    print(f"  '{word}'  bbox=({x0:.1f},{y0:.1f},{x1:.1f},{y1:.1f})")
print(f"  총 토큰 수: {len(blocks2)}")
print(f"  페이지 크기: {page2.rect.width:.0f} x {page2.rect.height:.0f} pt")
doc2.close()

print()

# ── GT와 매핑: 합성 첫 샘플 ──
gt = json.load(open("data/ground_truth/all_synth.json"))
print("=== GT 첫 샘플 invoice_number:", gt[0]["y_true"]["invoice_number"], "===")
# synth PDF 첫 페이지에서 해당 invoice_number 토큰 찾기
inv_no = gt[0]["y_true"]["invoice_number"]
doc3 = fitz.open("data/input/synth/all_synth.pdf")
page3 = doc3[0]
blocks3 = page3.get_text("words")
found = [(b[4], b[:4]) for b in blocks3 if inv_no in b[4] or b[4] in inv_no]
print("  invoice_number 매칭 토큰:", found)
doc3.close()
