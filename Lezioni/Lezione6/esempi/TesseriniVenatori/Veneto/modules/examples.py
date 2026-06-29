"""
examples.py
===========
Esempi d'uso completi della pipeline di elaborazione immagini.

Esegui con:
    python examples.py [percorso_immagine]
"""

from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np

# Importa la pipeline e le classi parametri
from pipeline import Pipeline
from filters import (
    GaussianBlurParams, CannyParams, BrightnessContrastParams,
    CLAHEParams, HSVAdjustParams, ResizeParams
)


# ---------------------------------------------------------------------------
# Helper: crea un'immagine di test se non viene fornita una reale
# ---------------------------------------------------------------------------

def make_test_image(h: int = 400, w: int = 600) -> np.ndarray:
    """Genera un'immagine BGR colorata per testare i filtri."""
    np.random.seed(0)
    img = np.zeros((h, w, 3), dtype=np.uint8)
    # Gradiente di sfondo
    for i in range(h):
        img[i, :, 2] = int(i / h * 200)  # canale R
        img[i, :, 0] = int((1 - i / h) * 150)  # canale B
    # Forme colorate
    cv2.rectangle(img, (50, 50), (250, 200), (0, 200, 100), -1)
    cv2.circle(img, (400, 150), 100, (200, 50, 220), -1)
    cv2.putText(img, "Pipeline Test", (60, 350),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
    # Rumore
    noise = np.random.randint(0, 40, img.shape, dtype=np.uint8)
    img = cv2.add(img, noise)
    return img


# ---------------------------------------------------------------------------
# Esempi
# ---------------------------------------------------------------------------

def esempio_1_fluent_api(img: np.ndarray) -> np.ndarray:
    """
    Fluent API: costruisci e applica la pipeline in una sola catena.
    Ogni metodo col nome del filtro è generato automaticamente.
    """
    print("\n── Esempio 1: Fluent API ──────────────────────────────────────")

    result = (
        Pipeline(img)
        .gaussian_blur(ksize=5)
        .brightness_contrast(alpha=1.2, beta=15)
        .sharpen(strength=0.7)
        .run()          # → np.ndarray BGR
    )
    print(f"  Output shape: {result.shape}, dtype: {result.dtype}")
    return result


def esempio_2_pipeline_riutilizzabile(img: np.ndarray) -> None:
    """
    Costruisci la pipeline una volta sola e applicala a più immagini.
    """
    print("\n── Esempio 2: Pipeline riutilizzabile ────────────────────────")

    pipe = (
        Pipeline()
        .clahe(clip_limit=3.0, tile_grid_size=8)
        .hsv_adjust(hue_shift=10, sat_scale=1.3)
        .resize(scale=0.5)
    )
    print(pipe)

    # Applica a immagini diverse (anche da array in memoria)
    out_a = pipe.run(img)
    out_b = pipe.run(img[:200, :300])   # crop diverso
    print(f"  out_a shape: {out_a.shape}")
    print(f"  out_b shape: {out_b.shape}")


def esempio_3_add_step_per_nome(img: np.ndarray) -> np.ndarray:
    """
    Aggiunta di filtri tramite stringa (utile se il nome viene da config).
    """
    print("\n── Esempio 3: add_step per nome stringa ──────────────────────")

    config = [
        ("median_blur",         {"ksize": 3}),
        ("brightness_contrast", {"alpha": 1.0, "beta": -20}),
        ("canny",               {"threshold1": 60, "threshold2": 140}),
    ]

    pipe = Pipeline(img)
    for name, params in config:
        pipe.add_step(name, **params)

    result = pipe.run()
    print(pipe.describe())
    return result


def esempio_4_modifica_pipeline(img: np.ndarray) -> None:
    """
    Modifica dinamica dei passi dopo la creazione.
    """
    print("\n── Esempio 4: Modifica dinamica della pipeline ───────────────")

    pipe = (
        Pipeline(img)
        .gaussian_blur(ksize=7)
        .brightness_contrast(alpha=1.5, beta=0)
        .gamma(gamma=1.2)
    )
    print("Prima:", pipe)

    # Aggiorna solo il parametro alpha del passo [1]
    pipe.update_step(1, alpha=0.8, beta=-30)

    # Rimuovi il passo [2]
    pipe.remove_step(2)

    # Aggiungi un nuovo passo in fondo
    pipe.morphology(operation="close", ksize=5)

    print("Dopo:", pipe)


def esempio_5_output_multipli(img: np.ndarray, outdir: Path) -> None:
    """
    Lo stesso risultato salvato in formati diversi.
    """
    print("\n── Esempio 5: Output multipli ────────────────────────────────")

    pipe = (
        Pipeline(img)
        .bilateral_filter(d=9, sigma_color=80, sigma_space=80)
        .clahe(clip_limit=2.5)
    )

    # Salva come file
    pipe.save(outdir / "output.jpg")
    pipe.save(outdir / "output.png")
    pipe.save(outdir / "output.webp")

    # Ottieni come array NumPy
    arr: np.ndarray = pipe.run()
    print(f"  NumPy array: {arr.shape}, {arr.dtype}")

    # Ottieni come Pillow (se disponibile)
    try:
        pil_img = pipe.to_pil()
        print(f"  PIL.Image: {pil_img.size}, mode={pil_img.mode}")
    except ImportError:
        print("  Pillow non installato, salto conversione PIL.")

    # Ottieni come bytes PNG
    raw: bytes = pipe.to_bytes()
    print(f"  bytes PNG: {len(raw):,} bytes")


def esempio_6_input_da_pillow(outdir: Path) -> None:
    """
    Input da PIL.Image (se Pillow è disponibile).
    """
    print("\n── Esempio 6: Input da Pillow ────────────────────────────────")
    try:
        import PIL.Image
        pil_img = PIL.Image.fromarray(
            cv2.cvtColor(make_test_image(), cv2.COLOR_BGR2RGB)
        )
        result = (
            Pipeline(pil_img)
            .gaussian_blur(ksize=3)
            .run(to=PIL.Image.Image)
        )
        print(f"  Input PIL → Output PIL: {result.size}")
    except ImportError:
        print("  Pillow non disponibile.")


def esempio_7_clona_e_varia(img: np.ndarray) -> None:
    """
    Clona una pipeline base e crea varianti con parametri diversi.
    """
    print("\n── Esempio 7: Clone e varianti ───────────────────────────────")

    base = Pipeline(img).gaussian_blur(ksize=5).clahe(clip_limit=2.0)

    varianti = [
        base.clone().update_step(1, clip_limit=1.0),
        base.clone().update_step(1, clip_limit=4.0),
        base.clone().update_step(1, clip_limit=8.0),
    ]

    for i, v in enumerate(varianti):
        out = v.run()
        print(f"  Variante {i}: mean brightness = {out.mean():.1f}")


def esempio_8_inspect(img: np.ndarray) -> None:
    """
    Ispezione: descrizione testuale + filtri disponibili.
    """
    print("\n── Esempio 8: Ispezione ──────────────────────────────────────")

    pipe = Pipeline(img).sharpen(strength=1.2).canny(threshold1=80, threshold2=160)
    print(pipe.describe())

    print("\n  Filtri disponibili:")
    for fname, info in Pipeline.available_filters().items():
        doc = info["doc"].splitlines()[0] if info["doc"] else ""
        print(f"    {fname:25s} – {doc}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Carica immagine reale o genera demo
    if len(sys.argv) > 1:
        import cv2
        src = cv2.imread(sys.argv[1])
        if src is None:
            print(f"Impossibile aprire {sys.argv[1]}, uso immagine demo.")
            src = make_test_image()
    else:
        print("Uso immagine demo (passa un percorso come argomento per usarne una tua).")
        src = make_test_image()

    outdir = Path("out")
    outdir.mkdir(exist_ok=True)

    # Esegui tutti gli esempi
    esempio_1_fluent_api(src)
    esempio_2_pipeline_riutilizzabile(src)
    esempio_3_add_step_per_nome(src)
    esempio_4_modifica_pipeline(src)
    esempio_5_output_multipli(src, outdir)
    esempio_6_input_da_pillow(outdir)
    esempio_7_clona_e_varia(src)
    esempio_8_inspect(src)

    print(f"\n✓ Output salvati in: {outdir.resolve()}")