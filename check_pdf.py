import subprocess, sys

# Try PyMuPDF first
try:
    import fitz
    files = [
        "data/input/invoices_val.pdf",
        "data/input/invoices_test.pdf",
        "data/input/all_invoices_.pdf",
        "data/input/synth/all_synth.pdf",
        "data/input/synth/train_synth.pdf",
        "data/input/synth/val_synth.pdf",
    ]
    for f in files:
        try:
            doc = fitz.open(f)
            print(f"{f} -> {doc.page_count} pages")
            doc.close()
        except Exception as e:
            print(f"{f} -> ERROR: {e}")
except ImportError:
    # Try pikepdf
    try:
        import pikepdf
        files = [
            "data/input/invoices_val.pdf",
            "data/input/invoices_test.pdf",
            "data/input/all_invoices_.pdf",
            "data/input/synth/all_synth.pdf",
            "data/input/synth/train_synth.pdf",
            "data/input/synth/val_synth.pdf",
        ]
        for f in files:
            try:
                pdf = pikepdf.open(f)
                print(f"{f} -> {len(pdf.pages)} pages")
                pdf.close()
            except Exception as e:
                print(f"{f} -> ERROR: {e}")
    except ImportError:
        print("Neither fitz nor pikepdf available")
        print("Try: pip install pymupdf  OR  pip install pikepdf")
