"""
pipeline.py
===========
API pubblica della pipeline di elaborazione immagini.

Uso rapido
----------
::

    from pipeline import Pipeline
    from filters import GaussianBlurParams, CannyParams

    result = (
        Pipeline("foto.jpg")
        .gaussian_blur(ksize=9)
        .brightness_contrast(alpha=1.3, beta=10)
        .canny(threshold1=80, threshold2=160)
        .save("output.png")          # salva su disco e ritorna se stesso
        .to_pil()                    # restituisce PIL.Image
    )

    # Oppure come builder riutilizzabile
    pipe = (
        Pipeline()
        .gaussian_blur(ksize=5)
        .sharpen(strength=0.8)
    )
    img_a = pipe.run("a.jpg")          # → np.ndarray BGR
    img_b = pipe.run("b.jpg", to=PIL.Image.Image)  # → PIL.Image
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .image_io import ImageInput, ImageOutput, read, write
from .filters import FILTER_REGISTRY, BaseParams


# ---------------------------------------------------------------------------
# Passo interno della pipeline
# ---------------------------------------------------------------------------

class _Step:
    """Rappresenta un singolo filtro con i suoi parametri."""

    def __init__(self, name: str, params: BaseParams):
        self.name = name
        self.params = params

    def __repr__(self) -> str:
        fields = {
            f.name: getattr(self.params, f.name)
            for f in self.params.__dataclass_fields__.values()  # type: ignore[union-attr]
        }
        args = ", ".join(f"{k}={v!r}" for k, v in fields.items())
        return f"{self.name}({args})"

    def apply(self, arr: np.ndarray) -> np.ndarray:
        _, fn = FILTER_REGISTRY[self.name]
        return fn(arr, self.params)


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class Pipeline:
    """
    Pipeline componibile di filtri OpenCV.

    Parameters
    ----------
    source:
        Immagine sorgente opzionale. Se fornita, viene caricata subito.
        Può essere passata anche in seguito tramite ``load()`` o ``run()``.
    copy_on_run:
        Se True (default), ogni chiamata a ``run()`` lavora su una copia
        dell'immagine originale lasciando invariato lo stato interno.
    """

    def __init__(
        self,
        source: Optional[ImageInput] = None,
        *,
        copy_on_run: bool = True,
    ):
        self._steps: List[_Step] = []
        self._image: Optional[np.ndarray] = None
        self.copy_on_run = copy_on_run

        if source is not None:
            self.load(source)

    # ── I/O ─────────────────────────────────────────────────────────────────

    def load(self, source: ImageInput) -> "Pipeline":
        """Carica un'immagine come punto di partenza della pipeline."""
        self._image = read(source)
        return self

    def run(
        self,
        source: Optional[ImageInput] = None,
        *,
        to: Optional[ImageOutput] = None,
    ) -> Any:
        """
        Esegue tutti i passi della pipeline sull'immagine corrente (o su
        ``source`` se fornito) e restituisce il risultato.

        Parameters
        ----------
        source:
            Se fornito, sovrascrive l'immagine caricata con ``load()``.
        to:
            Destinazione dell'output.
            - ``None``             → restituisce ``np.ndarray`` BGR
            - ``str`` / ``Path``   → salva su disco e restituisce il path
            - ``np.ndarray``       → restituisce una copia dell'array
            - ``PIL.Image.Image``  → restituisce oggetto Pillow
            - ``bytes``            → restituisce bytes PNG
        """
        if source is not None:
            arr = read(source)
        elif self._image is not None:
            arr = self._image.copy() if self.copy_on_run else self._image
        else:
            raise RuntimeError(
                "Nessuna immagine caricata. Usa Pipeline(source) oppure .load(source)."
            )

        for step in self._steps:
            arr = step.apply(arr)

        if to is None:
            return arr
        return write(arr, to)

    def save(self, path: Union[str, Path], **run_kwargs) -> "Pipeline":
        """Scorciatoia: esegue la pipeline e salva su disco, poi ritorna se stesso."""
        self.run(to=path, **run_kwargs)
        return self

    def to_pil(self):
        """Scorciatoia: esegue e restituisce PIL.Image."""
        return self.run(to="PIL")

    def to_bytes(self) -> bytes:
        """Scorciatoia: esegue e restituisce bytes PNG."""
        return self.run(to=bytes)

    # ── Gestione passi ───────────────────────────────────────────────────────

    def add_step(self, filter_name: str, **kwargs) -> "Pipeline":
        """
        Aggiunge un passo alla pipeline per nome.

        Parameters
        ----------
        filter_name:
            Nome del filtro (chiave di ``FILTER_REGISTRY``).
        **kwargs:
            Parametri da sovrascrivere rispetto ai default.
        """
        if filter_name not in FILTER_REGISTRY:
            raise KeyError(
                f"Filtro sconosciuto: {filter_name!r}. "
                f"Disponibili: {list(FILTER_REGISTRY)}"
            )
        ParamsClass, _ = FILTER_REGISTRY[filter_name]
        params = ParamsClass(**kwargs).validate()
        self._steps.append(_Step(filter_name, params))
        return self

    def remove_step(self, index: int) -> "Pipeline":
        """Rimuove il passo all'indice specificato."""
        del self._steps[index]
        return self

    def replace_step(self, index: int, filter_name: str, **kwargs) -> "Pipeline":
        """Sostituisce il passo all'indice con un nuovo filtro."""
        if filter_name not in FILTER_REGISTRY:
            raise KeyError(f"Filtro sconosciuto: {filter_name!r}")
        ParamsClass, _ = FILTER_REGISTRY[filter_name]
        params = ParamsClass(**kwargs).validate()
        self._steps[index] = _Step(filter_name, params)
        return self

    def update_step(self, index: int, **kwargs) -> "Pipeline":
        """Aggiorna solo i parametri del passo all'indice specificato."""
        step = self._steps[index]
        step.params = step.params.replace(**kwargs)
        return self

    def clear_steps(self) -> "Pipeline":
        """Rimuove tutti i passi."""
        self._steps.clear()
        return self

    def clone(self) -> "Pipeline":
        """Restituisce una copia profonda della pipeline (passi + immagine)."""
        new = Pipeline(copy_on_run=self.copy_on_run)
        new._steps = copy.deepcopy(self._steps)
        if self._image is not None:
            new._image = self._image.copy()
        return new

    # ── Metodi ergonomici per ogni filtro ────────────────────────────────────
    # Generati automaticamente dai nomi in FILTER_REGISTRY

    def __getattr__(self, name: str):
        """
        Permette di chiamare direttamente ``pipeline.gaussian_blur(ksize=7)``
        invece di ``pipeline.add_step("gaussian_blur", ksize=7)``.
        """
        if name in FILTER_REGISTRY:
            def _method(**kwargs) -> "Pipeline":
                return self.add_step(name, **kwargs)
            _method.__name__ = name
            _method.__doc__ = (
                f"Aggiunge il filtro '{name}' alla pipeline.\n\n"
                f"Parametri:\n{FILTER_REGISTRY[name][0].__doc__ or ''}"
            )
            return _method
        raise AttributeError(f"'Pipeline' non ha attributo '{name}'")

    # ── Ispezione ────────────────────────────────────────────────────────────

    def __len__(self) -> int:
        return len(self._steps)

    def __repr__(self) -> str:
        has_img = self._image is not None
        steps_repr = "\n  ".join(str(s) for s in self._steps) or "(nessun passo)"
        return (
            f"Pipeline(image_loaded={has_img}, steps={len(self._steps)})\n"
            f"  {steps_repr}"
        )

    def describe(self) -> str:
        """Descrizione testuale della pipeline con range dei parametri."""
        lines = [f"Pipeline – {len(self._steps)} passi\n" + "─" * 50]
        for i, step in enumerate(self._steps):
            lines.append(f"[{i}] {step.name}")
            for fname, val in vars(step.params).items():
                if fname.startswith("_"):
                    continue
                rng = step.params._ranges.get(fname)
                rng_str = f"  range={rng}" if rng else ""
                lines.append(f"     {fname} = {val!r}{rng_str}")
        return "\n".join(lines)

    @staticmethod
    def available_filters() -> Dict[str, Dict]:
        """
        Restituisce un dizionario con tutti i filtri disponibili e i loro
        parametri (default + range).
        """
        result: Dict[str, Dict] = {}
        for name, (ParamsClass, _) in FILTER_REGISTRY.items():
            defaults = {
                f.name: f.default
                for f in ParamsClass.__dataclass_fields__.values()  # type: ignore[union-attr]
                if not f.name.startswith("_")
            }
            result[name] = {
                "params": defaults,
                "ranges": ParamsClass._ranges,
                "doc": (ParamsClass.__doc__ or "").strip(),
            }
        return result