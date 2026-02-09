# pip install playwright reportlab pillow
# playwright install chromium

from __future__ import annotations

import math
import os
from contextlib import contextmanager
from io import BytesIO
from pathlib import Path

from playwright.sync_api import sync_playwright
from PIL import Image, ImageChops
from reportlab.pdfgen import canvas as rl_canvas
from reportlab.lib.utils import ImageReader


@contextmanager
def _launch_browser(width: int, height: int, device_scale_factor: float):
    with sync_playwright() as p:
        browser = p.chromium.launch(
            headless=True,
            args=[
                "--headless=new",
                "--enable-gpu",
                "--ignore-gpu-blocklist",
                "--use-gl=angle",
                "--use-angle=swiftshader",
            ],
        )
        context = browser.new_context(
            viewport={"width": width, "height": height},
            device_scale_factor=device_scale_factor,
        )
        page = context.new_page()
        try:
            yield page
        finally:
            context.close()
            browser.close()


def _set_plotly_camera(page, root_selector: str, *, distance: float | None):
    """
    Keep your existing directionality/camera logic:
    - orthographic
    - cube aspectmode
    - same "front-corner" eye computation as in your file
    """
    ok = page.evaluate(
        """
        async (args) => {
          const { rootSelector, distance, zScale, sceneIndex } = args;

          const root = document.querySelector(rootSelector);
          const plot = root
            ? (root.classList && root.classList.contains('js-plotly-plot') ? root
               : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
            : null;

          if (!plot || !window.Plotly) return false;

          const ids = (plot._fullLayout && plot._fullLayout._subplots && plot._fullLayout._subplots.gl3d)
                      || ['scene'];
          const id = ids[Math.min(Math.max(sceneIndex|0, 0), ids.length - 1)];

          const r = (distance && distance > 0) ? distance : 1.8;
          const k = (zScale && zScale > 0) ? zScale : 0.55;

          // Keep your existing orientation (theta=0)
          const theta = 0 * Math.PI / 180;

          const x0 = r;
          const y0 = r;
          const z0 = r * k;

          const x = x0 * Math.cos(theta) - y0 * Math.sin(theta);
          const y = x0 * Math.sin(theta) + y0 * Math.cos(theta);
          const z = z0;

          const camera = {
            eye:    { x, y, z },
            center: { x: 0, y: 0, z: 0 },
            up:     { x: 0, y: 0, z: 1 }
          };

          const update = {};
          update[`${id}.camera`]     = camera;
          update[`${id}.aspectmode`] = 'cube';
          update[`${id}.projection`] = { type: 'orthographic' };

          await Plotly.relayout(plot, update);
          await new Promise(rq => requestAnimationFrame(rq));
          await Plotly.relayout(plot, update);

          // Hide modebar/notifier (purely cosmetic)
          const styleId = '__html3d_to_pdf_style__';
          if (!document.getElementById(styleId)) {
            const st = document.createElement('style');
            st.id = styleId;
            st.textContent = `.modebar{display:none!important;} .plotly-notifier{display:none!important;}`;
            document.head.appendChild(st);
          }

          return true;
        }
        """,
        {"rootSelector": root_selector, "distance": distance, "zScale": 0.55, "sceneIndex": 0},
    )
    return bool(ok)


def _screenshot_element_png(page, selector: str) -> bytes:
    handle = page.query_selector(selector)
    if not handle:
        raise RuntimeError(f"Could not find element: {selector}")
    box = handle.bounding_box()
    if not box:
        raise RuntimeError("Could not read element bounding box for screenshot.")
    return page.screenshot(clip=box, type="png")


def _bg_rgb_from_corners(img_rgba: Image.Image) -> tuple[int, int, int]:
    w, h = img_rgba.size
    corners = [
        img_rgba.getpixel((0, 0)),
        img_rgba.getpixel((w - 1, 0)),
        img_rgba.getpixel((0, h - 1)),
        img_rgba.getpixel((w - 1, h - 1)),
    ]
    # median per channel (more robust than average)
    return tuple(sorted([c[i] for c in corners])[len(corners) // 2] for i in range(3))


def _ink_mask(img_rgba: Image.Image, *, threshold: int = 10) -> Image.Image:
    """
    Binary-ish mask of non-background pixels.
    Works for opaque or transparent backgrounds.
    """
    w, h = img_rgba.size
    bg_rgb = _bg_rgb_from_corners(img_rgba)

    rgb = img_rgba.convert("RGB")
    bg = Image.new("RGB", (w, h), bg_rgb)
    diff = ImageChops.difference(rgb, bg).convert("L")
    mask = diff.point(lambda p: 255 if p > threshold else 0)

    # also include alpha (in case bg is transparent)
    alpha = img_rgba.split()[-1]
    amask = alpha.point(lambda a: 255 if a > 0 else 0)

    return ImageChops.lighter(mask, amask)


def _trim_image(img_rgba: Image.Image, *, threshold: int = 10, pad: int = 2) -> Image.Image:
    mask = _ink_mask(img_rgba, threshold=threshold)
    bbox = mask.getbbox()
    if not bbox:
        return img_rgba
    l, t, r, b = bbox
    w, h = img_rgba.size
    l = max(l - pad, 0)
    t = max(t - pad, 0)
    r = min(r + pad, w)
    b = min(b + pad, h)
    return img_rgba.crop((l, t, r, b))


def _standardize_size(img_rgba: Image.Image, *, max_dim_px: int | None) -> Image.Image:
    """
    Standardize by scaling (preserves aspect ratio). If max_dim_px is None, no scaling.
    This does NOT change directionality; it only changes pixel size.
    """
    if not max_dim_px:
        return img_rgba
    w, h = img_rgba.size
    m = max(w, h)
    if m <= max_dim_px:
        return img_rgba
    scale = max_dim_px / float(m)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    return img_rgba.resize((nw, nh), Image.Resampling.LANCZOS)


def repack_colorbar_from_right(
    png_bytes: bytes,
    *,
    # detection / crop
    cb_search_width_px: int = 520,   # how wide a right-side slice to search for the colorbar+labels
    threshold: int = 10,
    pad: int = 2,
    # layout
    gap_px: int = 6,                # final gap between plot and colorbar
    # size standardization
    standardize_max_dim_px: int | None = 1800,
) -> bytes:
    """
    1) (Optional) standardize size
    2) Find tight bbox of the right-most content (colorbar + tick labels) inside a right-side slice
    3) Find tight bbox of the plot on the left (everything left of the colorbar slice bbox)
    4) Recompose plot + colorbar adjacent with gap_px
    5) Trim final outer whitespace
    """
    img = Image.open(BytesIO(png_bytes)).convert("RGBA")
    img = _standardize_size(img, max_dim_px=standardize_max_dim_px)

    w, h = img.size
    mask = _ink_mask(img, threshold=threshold)

    # --- 1) detect colorbar bbox in the right-most slice ---
    sw = min(cb_search_width_px, w)
    right_x0 = w - sw
    right_mask = mask.crop((right_x0, 0, w, h))
    rb = right_mask.getbbox()
    if not rb:
        # no colorbar found; just trim and return
        out = _trim_image(img, threshold=threshold, pad=pad)
        bio = BytesIO()
        out.save(bio, format="PNG")
        return bio.getvalue()

    cb_l, cb_t, cb_r, cb_b = rb
    cb_box = (right_x0 + cb_l, cb_t, right_x0 + cb_r, cb_b)

    # sanity: if the detected right block is extremely thin, ignore
    if (cb_box[2] - cb_box[0]) < 8 or (cb_box[3] - cb_box[1]) < 40:
        out = _trim_image(img, threshold=threshold, pad=pad)
        bio = BytesIO()
        out.save(bio, format="PNG")
        return bio.getvalue()

    # pad the colorbar bbox a touch
    cb_box = (
        max(cb_box[0] - pad, 0),
        max(cb_box[1] - pad, 0),
        min(cb_box[2] + pad, w),
        min(cb_box[3] + pad, h),
    )

    # --- 2) detect plot bbox from the left side (everything left of cb_box[0]) ---
    left_limit = max(1, cb_box[0])  # ensure non-empty crop
    left_mask = mask.crop((0, 0, left_limit, h))
    lb = left_mask.getbbox()
    if not lb:
        # fallback: use global bbox but clamp right edge
        gb = mask.getbbox()
        if not gb:
            out = _trim_image(img, threshold=threshold, pad=pad)
            bio = BytesIO()
            out.save(bio, format="PNG")
            return bio.getvalue()
        l, t, r, b = gb
        lb = (l, t, min(r, left_limit), b)

    pl, pt, pr, pb = lb
    plot_box = (
        max(pl - pad, 0),
        max(pt - pad, 0),
        min(pr + pad, left_limit),
        min(pb + pad, h),
    )

    plot_img = img.crop(plot_box)
    cb_img = img.crop(cb_box)

    # --- 3) recompose ---
    out_w = plot_img.width + gap_px + cb_img.width
    out_h = max(plot_img.height, cb_img.height)
    out = Image.new("RGBA", (out_w, out_h), (0, 0, 0, 0))

    plot_y = (out_h - plot_img.height) // 2
    cb_y = (out_h - cb_img.height) // 2

    out.paste(plot_img, (0, plot_y))
    out.paste(cb_img, (plot_img.width + gap_px, cb_y))

    # --- 4) trim final ---
    out = _trim_image(out, threshold=threshold, pad=pad)

    bio = BytesIO()
    out.save(bio, format="PNG")
    return bio.getvalue()


def _png_to_pdf(png_bytes: bytes, pdf_path: str):
    bio = BytesIO(png_bytes)
    img_reader = ImageReader(bio)
    w, h = img_reader.getSize()

    c = rl_canvas.Canvas(pdf_path, pagesize=(w, h))
    c.drawImage(img_reader, 0, 0, width=w, height=h, mask="auto")
    c.showPage()
    c.save()


def html_3d_to_pdf(
    output_pdf: str,
    *,
    html_path: str | os.PathLike,
    selector: str = ".js-plotly-plot",
    width: int = 1280,
    height: int = 960,
    device_scale_factor: float = 2.0,
    wait_ms_after_load: int = 500,
    load_timeout_ms: int = 60000,
    distance: float | None = 1.8,
    # postprocess controls
    standardize_max_dim_px: int | None = 1800,
    cb_search_width_px: int = 520,
    gap_px: int = 6,
) -> str:
    """
    Render Plotly HTML -> screenshot -> postprocess (cut & move colorbar) -> 1-page PDF
    """
    output_pdf = str(Path(output_pdf).resolve())
    html_uri = Path(html_path).resolve().as_uri()

    with _launch_browser(width, height, device_scale_factor) as page:
        page.goto(html_uri, wait_until="load", timeout=load_timeout_ms)
        page.wait_for_selector(selector, state="visible", timeout=load_timeout_ms)

        # Wait for Plotly full layout if possible
        try:
            page.wait_for_function(
                """
                (sel) => {
                  const root = document.querySelector(sel);
                  const plot = root
                    ? (root.classList && root.classList.contains('js-plotly-plot') ? root
                       : root.querySelector('.js-plotly-plot, .plotly-graph-div'))
                    : null;
                  return !!(plot && window.Plotly && plot._fullLayout);
                }
                """,
                arg=selector,
                timeout=load_timeout_ms,
            )
        except Exception:
            pass

        page.wait_for_timeout(wait_ms_after_load)

        # Set the camera (your established directionality)
        if not _set_plotly_camera(page, selector, distance=distance):
            raise RuntimeError(f"Could not set Plotly camera; check selector='{selector}'.")

        page.wait_for_timeout(250)

        # Screenshot -> postprocess -> write PDF
        png = _screenshot_element_png(page, selector)
        png = repack_colorbar_from_right(
            png,
            cb_search_width_px=cb_search_width_px,
            gap_px=gap_px,
            standardize_max_dim_px=standardize_max_dim_px,
        )
        _png_to_pdf(png, output_pdf)

    return output_pdf

if __name__ == "__main__":

    paths = [r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\affine3to5.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\helix.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\m1_sphere3.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\m9affine.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\M10_Cubic.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\M12_Norm.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\mn_nonlinear_3d.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\moebius.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\nonlinear4_6_8.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\nonlinear4to6.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\roll.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\scurve.html",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\interactive_datasets\uniform_3d.html"]

    names = [r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\affine3to5.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\helix.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\m1_sphere.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\m9affine.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\m10cubic.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\m12norm.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\mn_nonlinear_3d.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\moebius.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\nonlinear4_6_8.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\nonlinear4to6.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\roll.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\scurve.pdf",
    r"C:\Users\krp\PycharmProjects\FinalFixLIDGit\Bagging_for_LID\data_pdfs\uniform_3d.pdf"]

    for i in range(len(paths[0:2])):
        path = paths[i]
        name = names[i]

        pdf_path = html_3d_to_pdf(
            output_pdf=name,
            html_path=path,
            selector=".js-plotly-plot",
            distance=1.4,
            cb_search_width_px=1000,
            gap_px=4,
            standardize_max_dim_px=1800
        )