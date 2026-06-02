"""
image_io.py
===========
Adattatori di I/O per leggere e scrivere immagini da/verso
sorgenti e destinazioni eterogenee (file, Pillow, PyMuPDF,
OpenCV, NumPy, bytes grezzo).

Tutte le immagini all'interno della pipeline viaggiano come
``np.ndarray`` BGR uint8 – il formato nativo di OpenCV.
"""

from __future__ import annotations

import importlib
from importlib.util import find_spec
import io
import os
from pathlib import Path
from typing import Union

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Tipi accettati in ingresso
# ---------------------------------------------------------------------------
ImageInput = Union[
    str,            # percorso file
    Path,           # percorso file (pathlib)
    np.ndarray,     # array OpenCV (BGR) o RGB
    bytes,          # raw bytes di un file immagine
    "PIL.Image.Image",      # noqa: F821  – Pillow (opzionale)
    "fitz.Pixmap",          # noqa: F821  – PyMuPDF (opzionale)
    "fitz.Page",            # noqa: F821  – PyMuPDF page (opzionale)
]

ImageOutput = Union[
    str,    # percorso file  → salva su disco
    Path,   # percorso file  → salva su disco
    type,   # classe target  → converte e restituisce oggetto
]

# ---------------------------------------------------------------------------
# Helpers interni
# ---------------------------------------------------------------------------

def _is_available(module: str) -> bool:
    return find_spec(module) is not None


def _pil_to_bgr(pil_img) -> np.ndarray:
    import PIL.Image
    pil_img = pil_img.convert("RGB")
    return cv2.cvtColor(np.asarray(pil_img, dtype=np.uint8), cv2.COLOR_RGB2BGR)


def _bgr_to_pil(arr: np.ndarray):
    import PIL.Image
    return PIL.Image.fromarray(cv2.cvtColor(arr, cv2.COLOR_BGR2RGB))


def _fitz_pixmap_to_bgr(pixmap) -> np.ndarray:
    """Converte un fitz.Pixmap in array BGR."""
    import fitz  # noqa: F401
    # Forza spazio RGB (3 canali)
    if pixmap.n != 3:
        pixmap = fitz.Pixmap(fitz.csRGB, pixmap)
    arr = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width, 3
    )
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def _fitz_page_to_bgr(page, dpi: int = 150) -> np.ndarray:
    """Rasterizza una pagina PDF/XPS di PyMuPDF."""
    mat = page.fitz.Matrix(dpi / 72, dpi / 72)  # type: ignore[attr-defined]
    pixmap = page.get_pixmap(matrix=mat)
    return _fitz_pixmap_to_bgr(pixmap)


# ---------------------------------------------------------------------------
# Lettura
# ---------------------------------------------------------------------------

def read(source: ImageInput, *, flags: int = cv2.IMREAD_COLOR) -> np.ndarray:
    """
    Legge un'immagine da qualsiasi sorgente supportata e la restituisce
    come ``np.ndarray`` BGR uint8.

    Parameters
    ----------
    source:
        - ``str`` / ``Path``      → percorso file su disco
        - ``np.ndarray``          → già un array; verrà copiato e,
                                    se a 3 canali, si assume BGR
        - ``bytes``               → contenuto grezzo di un file immagine
        - ``PIL.Image.Image``     → immagine Pillow
        - ``fitz.Pixmap``         → pixmap PyMuPDF
        - ``fitz.Page``           → pagina PyMuPDF (rasterizzata a 150 dpi)
    flags:
        Flag di ``cv2.imread`` (default: ``cv2.IMREAD_COLOR``).
    """
    # ── File su disco ────────────────────────────────────────────────────────
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"File non trovato: {path}")
        arr = cv2.imread(str(path), flags)
        if arr is None:
            raise ValueError(f"OpenCV non riesce a leggere: {path}")
        return arr

    # ── NumPy array ──────────────────────────────────────────────────────────
    if isinstance(source, np.ndarray):
        arr = source.copy()
        # Gestisce array RGB (es. provenienti da matplotlib/skimage)
        # Euristica: se float → converti; se 3ch e non viene da cv2 è
        # probabile che sia RGB, ma non possiamo saperlo con certezza →
        # lasciamo la responsabilità al chiamante tramite il flag apposito.
        if arr.dtype != np.uint8:
            if arr.dtype in (np.float32, np.float64):
                arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)
        return arr

    # ── Bytes grezzi ─────────────────────────────────────────────────────────
    if isinstance(source, (bytes, bytearray)):
        buf = np.frombuffer(source, dtype=np.uint8)
        arr = cv2.imdecode(buf, flags)
        if arr is None:
            raise ValueError("Impossibile decodificare i bytes come immagine.")
        return arr

    # ── Pillow ───────────────────────────────────────────────────────────────
    if _is_available("PIL"):
        import PIL.Image
        if isinstance(source, PIL.Image.Image):
            return _pil_to_bgr(source)

    # ── PyMuPDF ──────────────────────────────────────────────────────────────
    if _is_available("fitz"):
        import fitz
        if isinstance(source, fitz.Pixmap):
            return _fitz_pixmap_to_bgr(source)
        if isinstance(source, fitz.Page):
            return _fitz_page_to_bgr(source)

    raise TypeError(
        f"Tipo sorgente non supportato: {type(source).__qualname__}. "
        "Valori accettati: str, Path, np.ndarray, bytes, "
        "PIL.Image.Image, fitz.Pixmap, fitz.Page."
    )


# ---------------------------------------------------------------------------
# Scrittura / conversione
# ---------------------------------------------------------------------------

def write(
    arr: np.ndarray,
    destination: ImageOutput,
    *,
    quality: int = 95,
    dpi: int = 150,
) -> object:
    """
    Scrive o converte un array BGR nell'output desiderato.

    Parameters
    ----------
    arr:
        Array BGR uint8 (output della pipeline).
    destination:
        - ``str`` / ``Path``      → salva su disco nel formato
                                    desunto dall'estensione
        - ``np.ndarray``          → restituisce una copia dell'array
        - ``PIL.Image.Image``     → converte e restituisce oggetto Pillow
        - ``"PIL"`` (stringa)     → alias per il tipo Pillow
        - ``bytes``               → restituisce i byte PNG dell'immagine
        - ``"bytes"`` (stringa)   → alias per il tipo bytes
    quality:
        Qualità JPEG (1–100).
    """
    # ── File su disco ────────────────────────────────────────────────────────
    if isinstance(destination, Path):
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        ext = path.suffix.lower()
        params: list[int] = []
        if ext in (".jpg", ".jpeg"):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif ext == ".png":
            compress = max(0, min(9, (100 - quality) // 11))
            params = [cv2.IMWRITE_PNG_COMPRESSION, compress]
        elif ext == ".webp":
            params = [cv2.IMWRITE_WEBP_QUALITY, quality]
        ok = cv2.imwrite(str(path), arr, params)
        if not ok:
            raise IOError(f"cv2.imwrite fallito per: {path}")
        return str(path)

    # ── NumPy array ──────────────────────────────────────────────────────────
    if destination is np.ndarray or destination == "ndarray":
        return arr.copy()

    # ── Pillow ───────────────────────────────────────────────────────────────
    pil_target = destination == "PIL" or (
        _is_available("PIL") and destination is __import__("PIL.Image", fromlist=["Image"]).Image
    )
    if pil_target or (isinstance(destination, str) and destination.upper() == "PIL"):
        return _bgr_to_pil(arr)

    if _is_available("PIL"):
        import PIL.Image
        if destination is PIL.Image.Image or (
            isinstance(destination, type) and issubclass(destination, PIL.Image.Image)
        ):
            return _bgr_to_pil(arr)

    # ── Bytes grezzi PNG ─────────────────────────────────────────────────────
    if destination is bytes or (isinstance(destination, str) and destination.lower() == "bytes"):
        ok, buf = cv2.imencode(".png", arr)
        if not ok:
            raise IOError("cv2.imencode fallito.")
        return buf.tobytes()

    raise TypeError(
        f"Destinazione non supportata: {destination!r}. "
        "Valori accettati: str/Path, np.ndarray, PIL.Image.Image, bytes."
    )