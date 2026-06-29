"""
filters.py
==========
Definizione di tutti i filtri disponibili nella pipeline.

Ogni filtro è rappresentato da:
  - una dataclass ``*Params`` che contiene i parametri con valori
    di default e i loro range ammessi (``_ranges``)
  - una funzione ``apply_<nome>(arr, params)`` che applica il filtro

L'elenco ``FILTER_REGISTRY`` mappa nome → (ParamsClass, apply_fn)
ed è la fonte unica di verità usata da ``Pipeline``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar, Dict, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------

@dataclass
class BaseParams:
    """
    Classe base per tutti i parametri di filtro.

    Sottoclassi devono definire ``_ranges``:
        dict[nome_campo → (min, max)] oppure (min, max, step) ecc.
    Il valore None su un lato dell'intervallo = nessun limite.
    """
    _ranges: ClassVar[Dict[str, Tuple]] = {}

    def validate(self) -> "BaseParams":
        """Verifica che tutti i campi rientrino nei rispettivi range."""
        for fname, rng in self._ranges.items():
            val = getattr(self, fname)
            if val is None:
                continue
            lo, hi = rng[0], rng[1]
            if lo is not None and val < lo:
                raise ValueError(f"{type(self).__name__}.{fname}={val} < min {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"{type(self).__name__}.{fname}={val} > max {hi}")
        return self

    def replace(self, **kwargs) -> "BaseParams":
        """Restituisce una copia con i campi sovrascritti e validata."""
        return dataclasses.replace(self, **kwargs).validate()  # type: ignore[arg-type]

    @classmethod
    def ranges(cls) -> Dict[str, Tuple]:
        return cls._ranges


# ---------------------------------------------------------------------------
# Filtri di sfocatura / smoothing
# ---------------------------------------------------------------------------

@dataclass
class GaussianBlurParams(BaseParams):
    """Sfocatura gaussiana."""
    ksize: int = 5           # dimensione kernel (dispari)
    sigma_x: float = 0.0    # deviazione std X (0 = auto)
    sigma_y: float = 0.0    # deviazione std Y (0 = auto)

    _ranges: ClassVar[Dict] = {
        "ksize":   (1, 101),
        "sigma_x": (0.0, 50.0),
        "sigma_y": (0.0, 50.0),
    }

    def validate(self) -> "GaussianBlurParams":
        super().validate()
        if self.ksize % 2 == 0:
            raise ValueError(f"ksize deve essere dispari, ricevuto {self.ksize}")
        return self


def apply_gaussian_blur(arr: np.ndarray, p: GaussianBlurParams) -> np.ndarray:
    return cv2.GaussianBlur(arr, (p.ksize, p.ksize), p.sigma_x, sigmaY=p.sigma_y)


@dataclass
class MedianBlurParams(BaseParams):
    """Sfocatura mediana (utile per rimozione salt-and-pepper)."""
    ksize: int = 5   # deve essere dispari ≥ 1

    _ranges: ClassVar[Dict] = {"ksize": (1, 51)}

    def validate(self) -> "MedianBlurParams":
        super().validate()
        if self.ksize % 2 == 0:
            raise ValueError(f"ksize deve essere dispari, ricevuto {self.ksize}")
        return self


def apply_median_blur(arr: np.ndarray, p: MedianBlurParams) -> np.ndarray:
    return cv2.medianBlur(arr, p.ksize)


@dataclass
class BilateralFilterParams(BaseParams):
    """Filtro bilaterale: sfuma preservando i bordi."""
    d: int = 9              # diametro pixel vicini
    sigma_color: float = 75.0
    sigma_space: float = 75.0

    _ranges: ClassVar[Dict] = {
        "d":           (1, 25),
        "sigma_color": (1.0, 250.0),
        "sigma_space": (1.0, 250.0),
    }


def apply_bilateral_filter(arr: np.ndarray, p: BilateralFilterParams) -> np.ndarray:
    return cv2.bilateralFilter(arr, p.d, p.sigma_color, p.sigma_space)


# ---------------------------------------------------------------------------
# Rilevamento bordi
# ---------------------------------------------------------------------------

@dataclass
class CannyParams(BaseParams):
    """Rilevamento bordi Canny."""
    threshold1: float = 100.0
    threshold2: float = 200.0
    aperture_size: int = 3   # 3, 5 oppure 7
    l2_gradient: bool = False

    _ranges: ClassVar[Dict] = {
        "threshold1":    (0.0, 1000.0),
        "threshold2":    (0.0, 1000.0),
        "aperture_size": (3, 7),
    }

    def validate(self) -> "CannyParams":
        super().validate()
        if self.aperture_size not in (3, 5, 7):
            raise ValueError("aperture_size deve essere 3, 5 o 7")
        return self


def apply_canny(arr: np.ndarray, p: CannyParams) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    edges = cv2.Canny(gray, p.threshold1, p.threshold2,
                      apertureSize=p.aperture_size, L2gradient=p.l2_gradient)
    return cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)


@dataclass
class SobelParams(BaseParams):
    """Gradiente di Sobel."""
    dx: int = 1
    dy: int = 1
    ksize: int = 3
    scale: float = 1.0
    delta: float = 0.0

    _ranges: ClassVar[Dict] = {
        "dx":    (0, 1),
        "dy":    (0, 1),
        "ksize": (1, 7),
        "scale": (0.0, 10.0),
        "delta": (-255.0, 255.0),
    }

    def validate(self) -> "SobelParams":
        super().validate()
        if self.dx == 0 and self.dy == 0:
            raise ValueError("dx e dy non possono essere entrambi 0")
        if self.ksize % 2 == 0:
            raise ValueError("ksize deve essere dispari")
        return self


def apply_sobel(arr: np.ndarray, p: SobelParams) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    sob = cv2.Sobel(gray.astype(np.float32), cv2.CV_32F,
                    p.dx, p.dy, ksize=p.ksize, scale=p.scale, delta=p.delta)
    sob = np.clip(np.abs(sob), 0, 255).astype(np.uint8)
    return cv2.cvtColor(sob, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Morfologia
# ---------------------------------------------------------------------------

@dataclass
class MorphologyParams(BaseParams):
    """Operazioni morfologiche (erosione, dilatazione, apertura, chiusura…)."""
    operation: str = "dilate"   # erode | dilate | open | close | gradient | tophat | blackhat
    kernel_shape: str = "rect"  # rect | ellipse | cross
    ksize: int = 5
    iterations: int = 1

    _ranges: ClassVar[Dict] = {
        "ksize":      (1, 51),
        "iterations": (1, 20),
    }

    _OPERATIONS: ClassVar[Dict[str, int]] = {
        "erode":    cv2.MORPH_ERODE,
        "dilate":   cv2.MORPH_DILATE,
        "open":     cv2.MORPH_OPEN,
        "close":    cv2.MORPH_CLOSE,
        "gradient": cv2.MORPH_GRADIENT,
        "tophat":   cv2.MORPH_TOPHAT,
        "blackhat": cv2.MORPH_BLACKHAT,
    }
    _SHAPES: ClassVar[Dict[str, int]] = {
        "rect":    cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE,
        "cross":   cv2.MORPH_CROSS,
    }

    def validate(self) -> "MorphologyParams":
        super().validate()
        if self.operation not in self._OPERATIONS:
            raise ValueError(f"operation non valida: {self.operation!r}. "
                             f"Valori: {list(self._OPERATIONS)}")
        if self.kernel_shape not in self._SHAPES:
            raise ValueError(f"kernel_shape non valido: {self.kernel_shape!r}. "
                             f"Valori: {list(self._SHAPES)}")
        return self


def apply_morphology(arr: np.ndarray, p: MorphologyParams) -> np.ndarray:
    shape = MorphologyParams._SHAPES[p.kernel_shape]
    op    = MorphologyParams._OPERATIONS[p.operation]
    kernel = cv2.getStructuringElement(shape, (p.ksize, p.ksize))
    return cv2.morphologyEx(arr, op, kernel, iterations=p.iterations)


# ---------------------------------------------------------------------------
# Sogliatura (Threshold)
# ---------------------------------------------------------------------------

@dataclass
class ThresholdParams(BaseParams):
    """Sogliatura con diverse modalità."""
    thresh: float = 127.0
    max_val: float = 255.0
    method: str = "binary"   # binary | binary_inv | trunc | tozero | tozero_inv | otsu | adaptive_mean | adaptive_gaussian

    _ranges: ClassVar[Dict] = {
        "thresh":  (0.0, 255.0),
        "max_val": (0.0, 255.0),
    }

    _METHODS: ClassVar[Dict[str, int]] = {
        "binary":      cv2.THRESH_BINARY,
        "binary_inv":  cv2.THRESH_BINARY_INV,
        "trunc":       cv2.THRESH_TRUNC,
        "tozero":      cv2.THRESH_TOZERO,
        "tozero_inv":  cv2.THRESH_TOZERO_INV,
        "otsu":        cv2.THRESH_BINARY | cv2.THRESH_OTSU,
    }

    def validate(self) -> "ThresholdParams":
        super().validate()
        valid = list(self._METHODS) + ["adaptive_mean", "adaptive_gaussian"]
        if self.method not in valid:
            raise ValueError(f"method non valido: {self.method!r}. Valori: {valid}")
        return self


def apply_threshold(arr: np.ndarray, p: ThresholdParams) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    if p.method == "adaptive_mean":
        out = cv2.adaptiveThreshold(gray, int(p.max_val),
                                    cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    elif p.method == "adaptive_gaussian":
        out = cv2.adaptiveThreshold(gray, int(p.max_val),
                                    cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    else:
        flag = ThresholdParams._METHODS[p.method]
        _, out = cv2.threshold(gray, p.thresh, p.max_val, flag)
    return cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Contrasto / Luminosità
# ---------------------------------------------------------------------------

@dataclass
class BrightnessContrastParams(BaseParams):
    """Regolazione lineare luminosità/contrasto: out = alpha*in + beta."""
    alpha: float = 1.0   # contrasto   [0.0 – 5.0]
    beta: float = 0.0    # luminosità  [-255 – 255]

    _ranges: ClassVar[Dict] = {
        "alpha": (0.0, 5.0),
        "beta":  (-255.0, 255.0),
    }


def apply_brightness_contrast(arr: np.ndarray, p: BrightnessContrastParams) -> np.ndarray:
    return cv2.convertScaleAbs(arr, alpha=p.alpha, beta=p.beta)


@dataclass
class CLAHEParams(BaseParams):
    """CLAHE – equalizzazione adattiva del contrasto (su canale L)."""
    clip_limit: float = 2.0
    tile_grid_size: int = 8   # griglia NxN

    _ranges: ClassVar[Dict] = {
        "clip_limit":     (0.5, 40.0),
        "tile_grid_size": (2, 64),
    }


def apply_clahe(arr: np.ndarray, p: CLAHEParams) -> np.ndarray:
    lab = cv2.cvtColor(arr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=p.clip_limit,
        tileGridSize=(p.tile_grid_size, p.tile_grid_size),
    )
    l_eq = clahe.apply(l)
    lab_eq = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Sharpening
# ---------------------------------------------------------------------------

@dataclass
class SharpenParams(BaseParams):
    """Sharpening tramite unsharp masking."""
    strength: float = 1.0    # intensità dell'effetto
    blur_ksize: int = 5

    _ranges: ClassVar[Dict] = {
        "strength":   (0.0, 5.0),
        "blur_ksize": (1, 51),
    }

    def validate(self) -> "SharpenParams":
        super().validate()
        if self.blur_ksize % 2 == 0:
            raise ValueError("blur_ksize deve essere dispari")
        return self


def apply_sharpen(arr: np.ndarray, p: SharpenParams) -> np.ndarray:
    blurred = cv2.GaussianBlur(arr, (p.blur_ksize, p.blur_ksize), 0)
    sharpened = cv2.addWeighted(arr, 1 + p.strength, blurred, -p.strength, 0)
    return sharpened


# ---------------------------------------------------------------------------
# Color space / HSV adjustments
# ---------------------------------------------------------------------------

@dataclass
class HSVAdjustParams(BaseParams):
    """Regolazione Hue, Saturation, Value nello spazio HSV."""
    hue_shift: int = 0          # spostamento tonalità [-180, 180]
    sat_scale: float = 1.0      # moltiplicatore saturazione
    val_scale: float = 1.0      # moltiplicatore luminosità

    _ranges: ClassVar[Dict] = {
        "hue_shift":  (-180, 180),
        "sat_scale":  (0.0, 5.0),
        "val_scale":  (0.0, 5.0),
    }


def apply_hsv_adjust(arr: np.ndarray, p: HSVAdjustParams) -> np.ndarray:
    hsv = cv2.cvtColor(arr, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 0] = (hsv[:, :, 0] + p.hue_shift) % 180
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * p.sat_scale, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * p.val_scale, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


# ---------------------------------------------------------------------------
# Resize / Warp
# ---------------------------------------------------------------------------

@dataclass
class ResizeParams(BaseParams):
    """Ridimensionamento proporzionale o assoluto."""
    width: Optional[int] = None    # None = mantieni aspetto
    height: Optional[int] = None
    scale: float = 1.0             # usato se width e height sono None
    interpolation: str = "linear"  # nearest | linear | cubic | area | lanczos

    _ranges: ClassVar[Dict] = {
        "scale": (0.01, 20.0),
    }

    _INTERP: ClassVar[Dict[str, int]] = {
        "nearest": cv2.INTER_NEAREST,
        "linear":  cv2.INTER_LINEAR,
        "cubic":   cv2.INTER_CUBIC,
        "area":    cv2.INTER_AREA,
        "lanczos": cv2.INTER_LANCZOS4,
    }

    def validate(self) -> "ResizeParams":
        super().validate()
        if self.interpolation not in self._INTERP:
            raise ValueError(f"interpolation non valida: {self.interpolation!r}")
        return self


def apply_resize(arr: np.ndarray, p: ResizeParams) -> np.ndarray:
    interp = ResizeParams._INTERP[p.interpolation]
    h, w = arr.shape[:2]
    if p.width is not None and p.height is not None:
        new_w, new_h = p.width, p.height
    elif p.width is not None:
        ratio = p.width / w
        new_w, new_h = p.width, int(h * ratio)
    elif p.height is not None:
        ratio = p.height / h
        new_w, new_h = int(w * ratio), p.height
    else:
        new_w, new_h = int(w * p.scale), int(h * p.scale)
    return cv2.resize(arr, (new_w, new_h), interpolation=interp)


@dataclass
class RotateParams(BaseParams):
    """Rotazione attorno al centro dell'immagine."""
    angle: float = 0.0       # gradi, antiorario
    scale: float = 1.0
    expand: bool = True      # se True adatta il canvas per non tagliare

    _ranges: ClassVar[Dict] = {
        "angle": (-360.0, 360.0),
        "scale": (0.01, 10.0),
    }


def apply_rotate(arr: np.ndarray, p: RotateParams) -> np.ndarray:
    h, w = arr.shape[:2]
    cx, cy = w / 2, h / 2
    M = cv2.getRotationMatrix2D((cx, cy), p.angle, p.scale)
    if p.expand:
        cos, sin = abs(M[0, 0]), abs(M[0, 1])
        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)
        M[0, 2] += (new_w - w) / 2
        M[1, 2] += (new_h - h) / 2
        w, h = new_w, new_h
    return cv2.warpAffine(arr, M, (w, h))


# ---------------------------------------------------------------------------
# Color LUT (lookup table)
# ---------------------------------------------------------------------------

@dataclass
class GammaParams(BaseParams):
    """Correzione gamma tramite LUT."""
    gamma: float = 1.0

    _ranges: ClassVar[Dict] = {"gamma": (0.05, 10.0)}


def apply_gamma(arr: np.ndarray, p: GammaParams) -> np.ndarray:
    inv_gamma = 1.0 / p.gamma
    table = np.array(
        [((i / 255.0) ** inv_gamma) * 255 for i in range(256)], dtype=np.uint8
    )
    return cv2.LUT(arr, table)


# ---------------------------------------------------------------------------
# Denoising
# ---------------------------------------------------------------------------

@dataclass
class DenoiseParams(BaseParams):
    """Rimozione rumore con fastNlMeansDenoisingColored."""
    h: float = 10.0           # filtro intensità luminanza
    h_color: float = 10.0     # filtro intensità crominanza
    template_window: int = 7  # dispari
    search_window: int = 21   # dispari

    _ranges: ClassVar[Dict] = {
        "h":               (1.0, 100.0),
        "h_color":         (1.0, 100.0),
        "template_window": (3, 21),
        "search_window":   (7, 63),
    }

    def validate(self) -> "DenoiseParams":
        super().validate()
        for f in ("template_window", "search_window"):
            if getattr(self, f) % 2 == 0:
                raise ValueError(f"{f} deve essere dispari")
        return self


def apply_denoise(arr: np.ndarray, p: DenoiseParams) -> np.ndarray:
    return cv2.fastNlMeansDenoisingColored(
        arr, None, p.h, p.h_color, p.template_window, p.search_window
    )


# ---------------------------------------------------------------------------
# Invert
# ---------------------------------------------------------------------------

@dataclass
class InvertParams(BaseParams):
    """Inversione bitwise dell'immagine (equivale a 255 - pixel)."""
    _ranges: ClassVar[Dict] = {}


def apply_invert(arr: np.ndarray, p: InvertParams) -> np.ndarray:
    return cv2.bitwise_not(arr)


# ---------------------------------------------------------------------------
# Grayscale
# ---------------------------------------------------------------------------

@dataclass
class GrayscaleParams(BaseParams):
    """Conversione in scala di grigi. L'output rimane a 3 canali (BGR) per compatibilità pipeline."""
    _ranges: ClassVar[Dict] = {}


def apply_grayscale(arr: np.ndarray, p: GrayscaleParams) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


# ---------------------------------------------------------------------------
# Deskew
# ---------------------------------------------------------------------------

@dataclass
class DeskewParams(BaseParams):
    """Correzione automatica dell'inclinazione (deskew) tramite minAreaRect sul testo."""
    min_angle: float = 0.5   # non corregge angoli inferiori a questa soglia (gradi)
    max_angle: float = 45.0  # non corregge angoli superiori a questa soglia (prob. falso positivo)

    _ranges: ClassVar[Dict] = {
        "min_angle": (0.0, 10.0),
        "max_angle": (1.0, 45.0),
    }


def _detect_skew_angle(gray: np.ndarray) -> float:
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) < 5:
        return 0.0
    angle = cv2.minAreaRect(coords)[-1]
    return -(90 + angle) if angle < -45 else -angle


def apply_deskew(arr: np.ndarray, p: DeskewParams) -> np.ndarray:
    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY) if arr.ndim == 3 else arr
    angle = _detect_skew_angle(gray)
    if abs(angle) < p.min_angle or abs(angle) > p.max_angle:
        return arr
    h, w = arr.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    return cv2.warpAffine(arr, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

FILTER_REGISTRY: Dict[str, Tuple[type, Callable]] = {
    "gaussian_blur":       (GaussianBlurParams,       apply_gaussian_blur),
    "median_blur":         (MedianBlurParams,          apply_median_blur),
    "bilateral_filter":    (BilateralFilterParams,     apply_bilateral_filter),
    "canny":               (CannyParams,               apply_canny),
    "sobel":               (SobelParams,               apply_sobel),
    "morphology":          (MorphologyParams,          apply_morphology),
    "threshold":           (ThresholdParams,           apply_threshold),
    "brightness_contrast": (BrightnessContrastParams,  apply_brightness_contrast),
    "clahe":               (CLAHEParams,               apply_clahe),
    "sharpen":             (SharpenParams,             apply_sharpen),
    "hsv_adjust":          (HSVAdjustParams,           apply_hsv_adjust),
    "resize":              (ResizeParams,              apply_resize),
    "rotate":              (RotateParams,              apply_rotate),
    "gamma":               (GammaParams,               apply_gamma),
    "denoise":             (DenoiseParams,             apply_denoise),
    "grayscale":           (GrayscaleParams,           apply_grayscale),
    "deskew":              (DeskewParams,              apply_deskew),
    "invert":              (InvertParams,              apply_invert),
}