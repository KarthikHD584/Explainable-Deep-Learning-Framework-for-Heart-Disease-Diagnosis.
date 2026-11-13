"""Fill the revised paper with the results of heart_xai_corrected_evaluation.ipynb.

Usage (Colab cell or terminal):
    python fill_paper.py Heart_Disease_XAI_IEEE_Conference_revised.docx RESULTS_DIR [output.docx]

RESULTS_DIR must contain paper_values.json, fig3_confusion.png and fig5_gradcam.png
(all written by the notebook). Only the Python standard library is needed.

What it does:
  * every yellow [[KEY]] in the paper is replaced by its value and the highlight is removed;
  * the placeholder images for Fig. 3 and Fig. 5 are replaced by the notebook's figures;
  * green [[AUTHOR: ...]] notes are left in place and listed, because they need your own words
    after you have looked at the Grad-CAM maps.
"""
import json, os, re, sys, zipfile
from xml.sax.saxutils import escape

src, res_dir = sys.argv[1], sys.argv[2]
out = sys.argv[3] if len(sys.argv) > 3 else src.replace(".docx", "_filled.docx")

values = json.load(open(os.path.join(res_dir, "paper_values.json"), encoding="utf-8"))
figures = {"word/media/fig3_confusion.png": os.path.join(res_dir, "fig3_confusion.png"),
           "word/media/fig5_gradcam.png": os.path.join(res_dir, "fig5_gradcam.png")}
for f in figures.values():
    if not os.path.exists(f):
        sys.exit(f"Missing {f}")

TOKEN_RUN = re.compile(
    r'(<w:r><w:rPr>(?:(?!</w:rPr>).)*?)<w:highlight w:val="yellow"/>((?:(?!</w:rPr>).)*?</w:rPr>'
    r'<w:t[^>]*>)\[\[([A-Z0-9_]+)\]\](</w:t></w:r>)', re.S)
missing = set()

def fill(m):
    key = m.group(3)
    if key not in values:
        missing.add(key)
        return m.group(0)
    return m.group(1) + m.group(2) + escape(values[key]) + m.group(4)

with zipfile.ZipFile(src) as zin, zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zout:
    for item in zin.infolist():
        data = zin.read(item.filename)
        if item.filename == "word/document.xml":
            xml = TOKEN_RUN.sub(fill, data.decode("utf-8"))
            left = sorted(set(re.findall(r"\[\[([A-Z0-9_]+)\]\]", xml)))
            notes = re.findall(r"\[\[AUTHOR:(.*?)\]\]", xml)
            data = xml.encode("utf-8")
        elif item.filename in figures:
            data = open(figures[item.filename], "rb").read()
        zout.writestr(item, data)

print(f"Wrote {out}")
print(f"Filled {len(values)} values.")
if missing or left:
    print("NOT filled (no value in paper_values.json):", ", ".join(sorted(missing | set(left))))
print(f"\n{len(notes)} green [[AUTHOR: ...]] notes still need your own text:")
for n in notes:
    print("  -", re.sub(r"<[^>]+>", "", n).strip()[:110])
print("\nAlso re-read every sentence that contains a filled number (yellow before) and check that the wording "
      "still fits the result, e.g. 'drop' in Section V-B if the patient-level accuracy were higher.")
