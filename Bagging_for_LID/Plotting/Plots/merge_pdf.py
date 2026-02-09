# pip install pymupdf
import math
from typing import Iterable, Optional, Tuple
from PyPDF2 import PdfReader, PdfWriter
import fitz
from pathlib import Path

def crop_pdf(
    pdf_path: str,
    *,
    trim_left: float = 0.05,
    trim_right: float = 0.05,
    trim_top: float = 0.05,
    trim_bottom: float = 0.05,
    overwrite: bool = True,
) -> str:

    input_path = Path(pdf_path)
    if not input_path.exists():
        raise FileNotFoundError(input_path)

    reader = PdfReader(str(input_path))
    writer = PdfWriter()

    for page in reader.pages:
        media_box = page.mediabox

        x0 = float(media_box.left)
        y0 = float(media_box.bottom)
        x1 = float(media_box.right)
        y1 = float(media_box.top)

        width = x1 - x0
        height = y1 - y0

        new_x0 = x0 + trim_left   * width
        new_x1 = x1 - trim_right  * width
        new_y0 = y0 + trim_bottom * height
        new_y1 = y1 - trim_top    * height

        page.mediabox.lower_left  = (new_x0, new_y0)
        page.mediabox.upper_right = (new_x1, new_y1)

        writer.add_page(page)

    if overwrite:
        output_path = input_path
    else:
        output_path = input_path.with_name(input_path.stem + "_cropped.pdf")

    with open(output_path, "wb") as f:
        writer.write(f)

    return str(output_path)


def merge_pdfs_grid_mupdf(
    pdf_paths: Iterable[str],
    output_path: str,
    *,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    padding: float = 0.0,
    margin: float = 0.0,
    order: str = "row",
    align: str = "center",
) -> Tuple[float, float]:
    paths = list(pdf_paths)
    if not paths:
        raise ValueError("No input PDFs provided.")

    first_doc = fitz.open(paths[0])
    try:
        first_page = first_doc[0]
        w0, h0 = first_page.rect.width, first_page.rect.height
    finally:
        first_doc.close()

    n = len(paths)
    if cols is None and rows is None:
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)
    elif cols is None:
        rows = int(rows)
        if rows <= 0: raise ValueError("rows must be positive.")
        cols = math.ceil(n / rows)
    elif rows is None:
        cols = int(cols)
        if cols <= 0: raise ValueError("cols must be positive.")
        rows = math.ceil(n / cols)
    else:
        cols, rows = int(cols), int(rows)
        if cols * rows < n:
            raise ValueError("cols × rows is smaller than number of PDFs.")

    cell_w, cell_h = w0, h0
    out_w = 2 * margin + cols * cell_w + (cols - 1) * padding
    out_h = 2 * margin + rows * cell_h + (rows - 1) * padding

    out_doc = fitz.open()
    out_page = out_doc.new_page(width=out_w, height=out_h)

    def rc_for(k: int) -> Tuple[int, int]:
        if order not in {"row", "col"}:
            raise ValueError("order must be 'row' or 'col'.")
        if order == "row":
            r, c = divmod(k, cols)
        else:
            c, r = divmod(k, rows)
        return r, c

    def place_rect(r: int, c: int) -> fitz.Rect:
        x0 = margin + c * (cell_w + padding)
        y0 = margin + r * (cell_h + padding)
        return fitz.Rect(x0, y0, x0 + cell_w, y0 + cell_h)

    def aligned_rect(dst: fitz.Rect, src_size: Tuple[float, float]) -> fitz.Rect:
        sw, sh = src_size
        scale = min(dst.width / sw, dst.height / sh)
        w = sw * scale
        h = sh * scale
        x = dst.x0
        y = dst.y0
        if "right" in align:
            x = dst.x1 - w
        elif "center" in align or "middle" in align:
            x = dst.x0 + (dst.width - w) / 2
        # vertical
        if "bottom" in align:
            y = dst.y1 - h
        elif "center" in align or "middle" in align:
            y = dst.y0 + (dst.height - h) / 2
        return fitz.Rect(x, y, x + w, y + h)

    for k, pth in enumerate(paths):
        r, c = rc_for(k)
        cell = place_rect(r, c)

        src_doc = fitz.open(pth)
        try:
            sp = src_doc[0]
            src_rect = sp.rect
            target = aligned_rect(cell, (src_rect.width, src_rect.height))
            out_page.show_pdf_page(target, src_doc, 0)
        finally:
            src_doc.close()

    out_doc.save(output_path)
    out_doc.close()
    return out_w, out_h


# Convenience wrappers
def merge_side_by_side_mupdf(pdf_paths, output_path, padding=0.0, margin=0.0):
    return merge_pdfs_grid_mupdf(pdf_paths, output_path, cols=len(list(pdf_paths)), rows=1,
                                 padding=padding, margin=margin, order="row")

if __name__ == "__main__":
    pdf_mle_weight_with_t = [r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_with_t_mle_weight_with_t_total_mse.pdf',
    r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_with_t_mle_weight_with_t_total_var.pdf',
                             r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_with_t_mle_weight_with_t_total_bias2.pdf',]

    pdf_mle_smooth = [r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_smooth_total_mse.pdf',
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_smooth_total_var.pdf',
                      r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_smooth_total_bias2.pdf',]

    pdf_mle_weight = [
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_weight_total_mse.pdf',
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_weight_total_var.pdf',
    r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_weight_total_bias2.pdf']

    pdf_mada_smooth = [r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_smooth_total_mse.pdf',
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_smooth_total_var.pdf',
                       r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_smooth_total_bias2.pdf']

    pdf_mada_weight = [
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_weight_total_mse.pdf',
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_weight_total_var.pdf',
    r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_weight_total_bias2.pdf',]

    pdf_tle_smooth = [
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_smooth_total_mse.pdf',
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_smooth_total_var.pdf',
    r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_smooth_total_bias2.pdf']

    pdf_tle_weight = [
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_weight_total_mse.pdf',
r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_weight_total_var.pdf',
    r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_weight_total_bias2.pdf',]

    pdf_weight_merged = [r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_weight_total_mse.pdf',
                         r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_weight_total_mse.pdf',
                         r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_weight_total_mse.pdf',
                         ]

    pdf_smooth_merged = [r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mle_smooth_total_mse.pdf',
                         r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_tle_smooth_total_mse.pdf',
                         r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\mergedresult_effectiveness_test_mada_smooth_total_mse.pdf',
                         ]

    pdf_lists = [pdf_weight_merged, pdf_smooth_merged]
    #pdf_mle_smooth, pdf_mle_weight, pdf_mada_smooth, pdf_mada_weight, pdf_tle_smooth, pdf_tle_weight,
    output_path_mle_smooth = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_effectiveness_test_mle_smooth_merged.pdf'

    output_path_mle_weight = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_effectiveness_test_mle_weight_merged.pdf'

    output_path_mada_smooth = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_effectiveness_test_mada_smooth_merged.pdf'

    output_path_mada_weight = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_effectiveness_test_mada_weight_merged.pdf'

    output_path_tle_smooth = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_effectiveness_test_tle_smooth_merged.pdf'

    output_path_tle_weight = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_effectiveness_test_tle_weight_merged.pdf'

    output_path7 = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_weight_merged.pdf'

    output_path8 = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_smooth_merged.pdf'

    output_path_mle_with_t = r'C:\Users\krp\OneDrive - Syddansk Universitet\PycharmProjects\LID1\LIDBagging2\plots\radar\merged_new\mergedresult_mle_weight_with_t_merged.pdf'
    #output_path_mle_smooth, output_path_mle_weight, output_path_mada_smooth, output_path_mada_weight, output_path_tle_smooth, output_path_tle_weight,
    output_path_lists = [output_path7, output_path8]

    for i in range(len(pdf_lists)):
        merge_side_by_side_mupdf(pdf_lists[i], output_path_lists[i], padding=0.0, margin=0.0)
    #merge_side_by_side_mupdf(pdf_mle_weight_with_t, output_path_mle_with_t, padding=0.0, margin=0.0)