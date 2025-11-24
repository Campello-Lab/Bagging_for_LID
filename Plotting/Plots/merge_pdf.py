# pip install pymupdf
import math
from typing import Iterable, Optional, Tuple
import fitz  # PyMuPDF


def merge_pdfs_grid_mupdf(
    pdf_paths: Iterable[str],
    output_path: str,
    *,
    cols: Optional[int] = None,
    rows: Optional[int] = None,
    padding: float = 0.0,   # space between tiles (points)
    margin: float = 0.0,    # outer margin (points)
    order: str = "row",     # "row" (L→R, top→bottom) or "col" (top→bottom, L→R)
    align: str = "center",  # how to place inside each cell if aspect ratios differ: "center", "top", "bottom", "left", "right"
) -> Tuple[float, float]:
    """
    Compose single-page PDFs into one PDF page arranged in a grid.
    Returns (out_width, out_height) in points (1 pt = 1/72").
    """
    paths = list(pdf_paths)
    if not paths:
        raise ValueError("No input PDFs provided.")

    # Probe first page size to propose a pleasant default cell size.
    # We'll still scale each input to the computed cell rect.
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

    # Output page size: tile the "prototype" size (w0×h0) in the grid.
    cell_w, cell_h = w0, h0
    out_w = 2 * margin + cols * cell_w + (cols - 1) * padding
    out_h = 2 * margin + rows * cell_h + (rows - 1) * padding

    # Create output doc + page
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
        # Note: PDF coords origin at top-left in PyMuPDF's page coordinate system (y increases downward),
        # but we can work in this system directly.
        x0 = margin + c * (cell_w + padding)
        y0 = margin + r * (cell_h + padding)
        return fitz.Rect(x0, y0, x0 + cell_w, y0 + cell_h)

    def aligned_rect(dst: fitz.Rect, src_size: Tuple[float, float]) -> fitz.Rect:
        sw, sh = src_size
        # scale to fit inside dst (preserve aspect)
        scale = min(dst.width / sw, dst.height / sh)
        w = sw * scale
        h = sh * scale
        # alignment inside the cell
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

    # Draw each source page into its cell
    for k, pth in enumerate(paths):
        r, c = rc_for(k)
        cell = place_rect(r, c)

        src_doc = fitz.open(pth)
        try:
            sp = src_doc[0]  # first (and only) page
            src_rect = sp.rect  # its natural size
            target = aligned_rect(cell, (src_rect.width, src_rect.height))

            # This call draws the full vector page as a Form XObject into the target rect
            out_page.show_pdf_page(target, src_doc, 0)  # pno=0
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