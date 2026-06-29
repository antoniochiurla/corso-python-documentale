"""
marks.py
========
Rilevamento di crocette (segni X) in immagini binarie di documenti.

Le crocette vengono identificate tramite analisi dei contorni combinata
con uno scoring basato sulle proiezioni diagonali: una X autentica ha
densità di inchiostro significativa su entrambe le diagonali del suo
bounding box.

Uso rapido
----------
::

    from modules.marks import detect_crosses, draw_crosses

    crosses = detect_crosses(handwritten_only, min_size=12, max_size=120)
    for c in crosses:
        print(c['center'], c['score'])

    img_annotated = draw_crosses(handwritten_only, crosses)
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


# ---------------------------------------------------------------------------
# Scoring diagonale
# ---------------------------------------------------------------------------

def _cross_score(roi: np.ndarray) -> float:
    """
    Restituisce uno score [0..1] che misura quanto una regione somiglia a una X.

    Criteri:
    1. Entrambe le strisce diagonali (TL→BR e TR→BL) devono avere alta
       copertura di inchiostro (lo score è il minimo dei due rapporti).
    2. Il punto di incrocio delle due strisce deve trovarsi nella fascia
       verticale centrale [25%–75%] dell'altezza del bounding box.
       Questo elimina lettere come "V" (incrocio in basso), "N"/"4"
       (incrocio fuori centro) pur tenendo le vere X (incrocio al centro).

    Parameters
    ----------
    roi : np.ndarray
        Immagine uint8 a canale singolo (o 3 canali con valori uniformi).
        I segni devono essere SCURI (0) su sfondo CHIARO (255).
    """
    if roi.ndim == 3:
        roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

    h, w = roi.shape
    if h < 6 or w < 6:
        return 0.0

    # Maschera binaria: 1 dove c'è inchiostro (pixel scuri)
    ink = (roi < 128).astype(np.float32)
    total_ink = ink.sum()
    if total_ink == 0:
        return 0.0

    # Coordinate normalizzate [0..1]
    y = np.linspace(0.0, 1.0, h).reshape(-1, 1)
    x = np.linspace(0.0, 1.0, w).reshape(1, -1)

    thickness = 0.28
    diag1_mask = np.abs(y - x) < thickness          # TL→BR
    diag2_mask = np.abs(y - (1.0 - x)) < thickness  # TR→BL

    diag1_coverage = (ink * diag1_mask).sum() / total_ink
    diag2_coverage = (ink * diag2_mask).sum() / total_ink

    # ── Verifica posizione verticale del punto di incrocio ──────────────────
    # Per una X: crossing_y ≈ 0.5 (centro).
    # Per "N", "4", "K": crossing_y > 0.65 o < 0.25.
    overlap = ink * diag1_mask * diag2_mask
    overlap_sum = overlap.sum()
    if overlap_sum > 0:
        crossing_y = float((overlap * y).sum() / overlap_sum)
        if not (0.25 <= crossing_y <= 0.75):
            return 0.0

    # ── Verifica inchiostro nel riquadro centrale ────────────────────────────
    # Una X ha il punto di incrocio fisico al centro del bbox → alta densità
    # nella regione [35%–65%] in entrambe le direzioni.
    # Una "V" ha i tratti che divergono verso l'esterno a metà altezza,
    # lasciando il centro completamente vuoto.
    cy_lo = int(h * 0.35);  cy_hi = max(cy_lo + 1, int(h * 0.65))
    cx_lo = int(w * 0.35);  cx_hi = max(cx_lo + 1, int(w * 0.65))
    center_density = ink[cy_lo:cy_hi, cx_lo:cx_hi].mean()
    if center_density < 0.15:
        return 0.0

    # Lo score è il minimo: entrambi i tratti devono essere presenti
    return float(min(diag1_coverage, diag2_coverage))


# ---------------------------------------------------------------------------
# API pubblica
# ---------------------------------------------------------------------------

def detect_crosses(
    handwritten_only: np.ndarray,
    *,
    min_size: int = 12,
    max_size: int = 120,
    min_aspect: float = 0.5,
    max_aspect: float = 2.0,
    max_density: float = 0.60,
    cross_threshold: float = 0.35,
    exclude_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Trova le crocette (X) in un'immagine binaria prodotta dalla pipeline.

    Parameters
    ----------
    handwritten_only : np.ndarray
        Immagine uint8 (canale singolo o BGR) con segni SCURI su sfondo CHIARO.
        Tipicamente l'output di ``subtract_empty_from_filled``.
    min_size : int
        Dimensione minima (larghezza E altezza) del bounding box in pixel.
    max_size : int
        Dimensione massima (larghezza O altezza) del bounding box in pixel.
    min_aspect, max_aspect : float
        Range accettabile del rapporto larghezza/altezza del bounding box.
        Una X tende ad avere aspect ratio vicino a 1.
    max_density : float
        Rapporto massimo tra area del contorno e area del bounding box [0..1].
        Una X autentica ha le 4 aree angolari vuote → density 0.25–0.55.
        Blobs solidi (quadratini di allineamento, lettere bold) hanno density > 0.65.
        Default: 0.60.
    cross_threshold : float
        Soglia minima di ``_cross_score`` per considerare un contorno una X.
        Valori tipici: 0.30 (sensibile) … 0.45 (preciso).
    exclude_regions : list of (x1, y1, x2, y2), optional
        Aree rettangolari da ignorare: le crocette il cui centro cade all'interno
        di uno qualsiasi di questi rettangoli vengono scartate.
        Utile per escludere zone del modulo dove si sa che non ci sono crocette
        (es. banner data/giorno che cambia ogni giorno e non viene rimosso
        dalla sottrazione template).

    Returns
    -------
    list of dict
        Ogni elemento ha le chiavi:
        - ``'bbox'``     : (x, y, w, h) bounding box in pixel
        - ``'center'``   : (cx, cy) centro del bounding box
        - ``'score'``    : float, punteggio diagonale [0..1]
        - ``'density'``  : float, rapporto area contorno / area bbox
    """
    if handwritten_only.ndim == 3:
        gray = cv2.cvtColor(handwritten_only, cv2.COLOR_BGR2GRAY)
    else:
        gray = handwritten_only.copy()

    # Inverti per avere i segni come bianco (richiesto da findContours)
    inverted = cv2.bitwise_not(gray)
    contours, _ = cv2.findContours(inverted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    results: List[Dict[str, Any]] = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)

        # Filtri dimensione
        if w < min_size or h < min_size:
            continue
        if w > max_size or h > max_size:
            continue

        # Filtro aspect ratio
        aspect = w / max(h, 1)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        # Filtro densità: blobs solidi (quadratini, lettere bold dense) vengono scartati.
        # Una X ha le 4 aree angolari del bounding box vuote → bassa densità.
        bbox_area = w * h
        contour_area = cv2.contourArea(cnt)
        density = contour_area / max(bbox_area, 1)
        if density > max_density:
            continue

        roi = gray[y : y + h, x : x + w]
        score = _cross_score(roi)

        if score >= cross_threshold:
            cx, cy = x + w // 2, y + h // 2
            if exclude_regions and any(
                x1 <= cx <= x2 and y1 <= cy <= y2
                for (x1, y1, x2, y2) in exclude_regions
            ):
                continue
            results.append(
                {
                    "bbox": (x, y, w, h),
                    "center": (cx, cy),
                    "score": round(score, 3),
                    "density": round(density, 3),
                }
            )

    # Ordina per score decrescente
    results.sort(key=lambda c: c["score"], reverse=True)
    return results


# ---------------------------------------------------------------------------
# Rilevamento celle del modulo e verifica segni
# ---------------------------------------------------------------------------

def _cluster(vals: List[int], gap: int = 8) -> List[int]:
    """Raggruppa valori vicini e restituisce il centroide di ogni gruppo."""
    vals = sorted(set(vals))
    if not vals:
        return []
    groups: List[List[int]] = [[vals[0]]]
    for v in vals[1:]:
        if v - groups[-1][-1] <= gap:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [int(np.mean(g)) for g in groups]


def detect_standalone_boxes(
    template: np.ndarray,
    *,
    min_size: int = 8,
    max_size: int = 60,
    max_template_ink: float = 0.15,
    y_min: int = 0,
    y_max: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """
    Trova i piccoli quadratini isolati (checkbox) nel template usando
    ``findContours`` con gerarchia: rileva contorni rettangolari con
    4 vertici e interno quasi vuoto.

    Complementa :func:`detect_boxes` per i checkbox che non fanno parte
    della griglia principale (es. VAGANTE □, APPOSTAMENTO □, FUORI REGIONE □).

    Returns
    -------
    list of dict ``{'bbox': (x, y, w, h), 'template_ink': float}``
    """
    if template.ndim == 3:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray = template.copy()

    _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)
    contours, hier = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []

    ink_mask = (gray < 128).astype(np.float32)
    results: List[Dict[str, Any]] = []
    seen: set = set()

    for i, (cnt, h) in enumerate(zip(contours, hier[0])):
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        # Accetta forme con 4-6 vertici (quadrati leggermente irregolari per scansione)
        if not (4 <= len(approx) <= 6):
            continue
        x, y, w, h_box = cv2.boundingRect(cnt)
        if not (min_size <= w <= max_size and min_size <= h_box <= max_size):
            continue
        if y < y_min or (y_max is not None and y > y_max):
            continue
        # Aspect ratio: quadrati e rettangoli piccoli
        aspect = w / max(h_box, 1)
        if not (0.4 <= aspect <= 2.5):
            continue
        # Inchiostro interno (escludendo il bordo)
        m = 2
        roi = ink_mask[y + m: y + h_box - m, x + m: x + w - m]
        if roi.size == 0:
            continue
        interior_ink = float(roi.mean())
        if interior_ink > max_template_ink:
            continue
        key = (x // 6, y // 6)
        if key in seen:
            continue
        seen.add(key)
        results.append({"bbox": (x, y, w, h_box), "template_ink": round(interior_ink, 4)})

    results.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return results


def detect_boxes(
    template: np.ndarray,
    *,
    line_length: int = 30,
    cluster_gap: int = 8,
    min_cell_size: int = 10,
    max_cell_width: int = 100,
    max_cell_height: int = 55,
    max_template_ink: float = 0.12,
) -> List[Dict[str, Any]]:
    """
    Rileva le celle vuote (compilabili) di un modulo stampato.

    Usa operazioni morfologiche per estrarre le linee orizzontali e
    verticali del template, trova le intersezioni e ricostruisce le
    celle. Scarta le celle che nel template contengono già testo stampato
    (densità di inchiostro > ``max_template_ink``).

    Parameters
    ----------
    template : np.ndarray
        Immagine del modulo VUOTO (BGR o grayscale, sfondo chiaro).
    line_length : int
        Lunghezza minima (pixel) delle linee morfologiche da rilevare.
    cluster_gap : int
        Distanza massima (pixel) tra incroci per considerarli lo stesso punto.
    min_cell_size : int
        Dimensione minima (larghezza e altezza) di una cella in pixel.
    max_cell_width : int
        Larghezza massima di una cella: celle più larghe (colonne testo)
        vengono scartate perché contengono nomi/etichette stampate.
    max_cell_height : int
        Altezza massima di una cella. Celle più alte sono separatori
        strutturali o sezioni header, non caselle da compilare.
    max_template_ink : float
        Densità massima di inchiostro nel template per considerare una cella
        "vuota" (compilabile). Celle con più inchiostro sono etichette/testo.

    Returns
    -------
    list of dict
        Ogni elemento: ``{'bbox': (x, y, w, h), 'template_ink': float}``
        ordinato per posizione (top→bottom, left→right).
    """
    if template.ndim == 3:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray = template.copy()

    _, bw = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY_INV)

    h_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (line_length, 1))
    v_ker = cv2.getStructuringElement(cv2.MORPH_RECT, (1, line_length))
    h_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, h_ker)
    v_lines = cv2.morphologyEx(bw, cv2.MORPH_OPEN, v_ker)

    inter = cv2.bitwise_and(h_lines, v_lines)
    inter = cv2.dilate(inter, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(inter, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    raw_pts: List[Tuple[int, int]] = []
    for cnt in contours:
        M = cv2.moments(cnt)
        if M["m00"] > 0:
            raw_pts.append((int(M["m01"] / M["m00"]), int(M["m10"] / M["m00"])))

    if len(raw_pts) < 4:
        return []

    ys = _cluster([r for r, _ in raw_pts], cluster_gap)
    xs = _cluster([c for _, c in raw_pts], cluster_gap)

    ink_mask = (gray < 128).astype(np.float32)

    cells: List[Dict[str, Any]] = []
    for i in range(len(ys) - 1):
        for j in range(len(xs) - 1):
            y0, y1 = ys[i], ys[i + 1]
            x0, x1 = xs[j], xs[j + 1]
            w, h = x1 - x0, y1 - y0
            if w < min_cell_size or h < min_cell_size:
                continue
            if w > max_cell_width or h > max_cell_height:
                continue
            roi_ink = ink_mask[y0:y1, x0:x1].mean()
            if roi_ink > max_template_ink:
                continue
            cells.append({
                "bbox": (x0, y0, w, h),
                "template_ink": round(float(roi_ink), 4),
            })

    cells.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    return cells


def detect_form_boxes(
    template: np.ndarray,
    *,
    min_size: int = 14,
    max_size: int = 100,
    min_aspect: float = 0.35,
    max_aspect: float = 2.8,
    max_template_ink: float = 0.07,
    threshold: int = 200,
    dedup_gap: int = 8,
    exclude_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Rileva tutti i riquadri compilabili nel modulo leggendo direttamente
    il template tramite analisi dei contorni.

    Sostituisce detect_boxes + detect_standalone_boxes con un unico passaggio:
    individua i rettangoli disegnati nel modulo filtrando per dimensione,
    forma e assenza di inchiostro interno (riquadri vuoti da compilare).

    Parameters
    ----------
    template : np.ndarray
        Immagine del modulo VUOTO (BGR o grayscale).
    min_size : int
        Dimensione minima (larghezza E altezza) in pixel. Default 14 cattura
        anche i piccoli checkbox (VAGANTE, APPOSTAMENTO, AATV, AFV ~16-17 px).
    max_size : int
        Dimensione massima per lato. Default 100 copre il riquadro di
        conferma in alto a destra (~58x50 px).
    min_aspect, max_aspect : float
        Range del rapporto larghezza/altezza accettato.
    max_template_ink : float
        Densità massima di inchiostro interno (gray<128). Default 0.07.
    threshold : int
        Soglia di binarizzazione per rilevare i bordi dei riquadri.
        Default 200 cattura anche i riquadri rosa/colorati chiari
        (PROV, ATC, CA) che in grayscale hanno valore ~196.
    dedup_gap : int
        Due contorni il cui angolo top-left cade nella stessa cella di
        dedup_gap×dedup_gap pixel vengono deduplicati (si tiene il primo).
    exclude_regions : list of (x1, y1, x2, y2), optional
        Riquadri il cui centro cade in queste aree vengono ignorati.

    Returns
    -------
    list of dict ``{'bbox': (x, y, w, h), 'template_ink': float}``
        Ordinato per posizione: top→bottom, left→right.
    """
    if template.ndim == 3:
        gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    else:
        gray = template.copy()

    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)
    ink_mask = (gray < 128).astype(np.float32)

    contours, hier = cv2.findContours(bw, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []

    results: List[Dict[str, Any]] = []

    for cnt in contours:
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.06 * peri, True)
        if not (4 <= len(approx) <= 6):
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if not (min_size <= w <= max_size and min_size <= h <= max_size):
            continue

        aspect = w / max(h, 1)
        if not (min_aspect <= aspect <= max_aspect):
            continue

        if exclude_regions:
            cx, cy = x + w // 2, y + h // 2
            if any(x1 <= cx <= x2 and y1 <= cy <= y2 for x1, y1, x2, y2 in exclude_regions):
                continue

        # Margine adattivo: più grande per celle grandi, riduce il bleed del bordo
        # e aumenta la sensibilità all'inchiostro interno (numeri stampati nelle celle).
        m = min(max(2, min(w, h) // 6), 5)
        roi = ink_mask[y + m: y + h - m, x + m: x + w - m]
        if roi.size == 0:
            continue
        interior_ink = float(roi.mean())
        if interior_ink > max_template_ink:
            continue

        results.append({"bbox": (x, y, w, h), "template_ink": round(interior_ink, 4)})

    # NMS: elimina duplicati con IoU > 0.3 (tiene il box più grande)
    results.sort(key=lambda b: b["bbox"][2] * b["bbox"][3], reverse=True)
    kept: List[Dict[str, Any]] = []
    for cand in results:
        cx, cy, cw, ch = cand["bbox"]
        cx2, cy2 = cx + cw, cy + ch
        dup = False
        for k in kept:
            kx, ky, kw, kh = k["bbox"]
            kx2, ky2 = kx + kw, ky + kh
            ix0, iy0 = max(cx, kx), max(cy, ky)
            ix1, iy1 = min(cx2, kx2), min(cy2, ky2)
            if ix1 > ix0 and iy1 > iy0:
                inter = (ix1 - ix0) * (iy1 - iy0)
                union = cw * ch + kw * kh - inter
                if inter / max(union, 1) > 0.3:
                    dup = True
                    break
        if not dup:
            kept.append(cand)

    kept.sort(key=lambda b: (b["bbox"][1] // 10, b["bbox"][0]))
    return kept


def find_corner_marks(
    image: np.ndarray,
    *,
    y_margin: int = 130,
    x_margin: int = 220,
    min_area: float = 500,
    max_area: float = 3000,
    threshold: int = 70,
) -> Optional[Dict[str, Tuple[int, int]]]:
    """
    Individua i 4 marcatori neri di allineamento agli angoli del modulo.

    Ogni marcatore è un rettangolo nero solido (~58×30 px) posizionato
    vicino all'angolo della pagina. Vengono usati come sistema di riferimento
    per trasformare le posizioni dei riquadri dal template al foglio compilato.

    I marcatori di destra si trovano a circa 185-196px dal bordo destro,
    quindi ``x_margin`` deve essere ≥ 220 per catturarli.
    Il barcode stampato in alto è a y≥140, quindi ``y_margin=130`` lo esclude.

    Returns
    -------
    dict ``{'tl': (cx,cy), 'tr': (cx,cy), 'bl': (cx,cy), 'br': (cx,cy)}``
    oppure ``None`` se uno o più marcatori non vengono trovati.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
    h, w = gray.shape
    _, bw = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY_INV)

    zones = {
        'tl': (0,          y_margin, 0,          x_margin),
        'tr': (0,          y_margin, w - x_margin, w      ),
        'bl': (h - y_margin, h,      0,          x_margin),
        'br': (h - y_margin, h,      w - x_margin, w      ),
    }
    def _valid_shape(cnt) -> bool:
        """I marcatori angolari sono rettangoli pieni ~58×30px; bordi e linee no."""
        rx, ry, rw, rh = cv2.boundingRect(cnt)
        if rw == 0 or rh == 0:
            return False
        aspect = rw / rh
        if aspect < 0.25 or aspect > 8.0:   # rifiuta bordi troppo allungati
            return False
        solidity = cv2.contourArea(cnt) / (rw * rh)
        return solidity > 0.35              # deve essere per lo più pieno

    result: Dict[str, Tuple[int, int]] = {}
    for name, (y0, y1, x0, x1) in zones.items():
        roi = bw[y0:y1, x0:x1]
        cnts, _ = cv2.findContours(roi, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        valid = [c for c in cnts
                 if min_area <= cv2.contourArea(c) <= max_area and _valid_shape(c)]
        if not valid:
            continue  # marcatore non trovato in questa zona
        best = max(valid, key=cv2.contourArea)
        rx, ry, rw, rh = cv2.boundingRect(best)
        result[name] = (x0 + rx + rw // 2, y0 + ry + rh // 2)

    if len(result) < 3:
        return None

    # Stima parallelogramma: TL + BR = TR + BL (diagonali si bisecano).
    # Usata SOLO per stimare angoli mancanti, non per sovrascrivere quelli trovati.
    _para: Dict[str, Tuple[str, str, str]] = {
        'tl': ('tr', 'bl', 'br'),
        'tr': ('tl', 'br', 'bl'),
        'bl': ('tl', 'br', 'tr'),
        'br': ('tr', 'bl', 'tl'),
    }
    for name, (k1, k2, k3) in _para.items():
        if name not in result and all(k in result for k in (k1, k2, k3)):
            r1, r2, r3 = result[k1], result[k2], result[k3]
            result[name] = (r1[0] + r2[0] - r3[0], r1[1] + r2[1] - r3[1])

    return result if len(result) == 4 else None


def transform_boxes(
    template_corners: Dict[str, Tuple[int, int]],
    form_corners: Dict[str, Tuple[int, int]],
    boxes: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Trasforma le posizioni dei riquadri dal sistema di coordinate del template
    a quello del foglio compilato, usando i 4 marcatori angolari come riferimento.

    Parameters
    ----------
    template_corners : dict
        Output di :func:`find_corner_marks` sul template vuoto.
    form_corners : dict
        Output di :func:`find_corner_marks` sul foglio compilato.
    boxes : list of dict
        Output di :func:`detect_form_boxes` (coordinate nel template).

    Returns
    -------
    list of dict
        Stessa struttura dell'input, con ``'bbox'`` nelle coordinate del foglio.
    """
    src = np.float32([
        template_corners['tl'], template_corners['tr'],
        template_corners['br'], template_corners['bl'],
    ])
    dst = np.float32([
        form_corners['tl'], form_corners['tr'],
        form_corners['br'], form_corners['bl'],
    ])
    H, _ = cv2.findHomography(src, dst)

    result = []
    for box in boxes:
        x, y, bw, bh = box['bbox']
        pts = np.float32([[x, y], [x+bw, y], [x+bw, y+bh], [x, y+bh]]).reshape(-1, 1, 2)
        pts_t = cv2.perspectiveTransform(pts, H).reshape(-1, 2)
        x0 = int(np.round(pts_t[:, 0].min()))
        y0 = int(np.round(pts_t[:, 1].min()))
        x1 = int(np.round(pts_t[:, 0].max()))
        y1 = int(np.round(pts_t[:, 1].max()))
        new_box = dict(box)
        new_box['bbox'] = (x0, y0, x1 - x0, y1 - y0)
        result.append(new_box)
    return result


def check_box_marks(
    handwritten_only: np.ndarray,
    boxes: List[Dict[str, Any]],
    *,
    ink_threshold: float = 0.05,
    uncertain_threshold: float = 0.0,
    margin: int = 2,
    exclude_regions: Optional[List[Tuple[int, int, int, int]]] = None,
) -> List[Dict[str, Any]]:
    """
    Per ogni cella rilevata da :func:`detect_boxes`, misura la densità
    di inchiostro in ``handwritten_only`` e determina se la cella è barrata.

    Parameters
    ----------
    handwritten_only : np.ndarray
        Immagine binaria (sfondo chiaro) con solo il testo manoscritto,
        output di ``subtract_empty_from_filled``.
    boxes : list of dict
        Output di :func:`detect_boxes`.
    ink_threshold : float
        Densità minima [0..1] per considerare la cella "segnata" (certo).
    uncertain_threshold : float
        Densità minima per considerare la cella "incerta" (segno debole).
        0.0 (default) disabilita la categoria incerta.
        Deve essere < ink_threshold per avere effetto.
    margin : int
        Pixel di margine interno da ignorare (esclude i bordi della cella
        che potrebbero contenere residui del bordo stampato).
    exclude_regions : list of (x1, y1, x2, y2), optional
        Celle il cui centro cade in queste aree vengono sempre marcate come
        ``checked=False``, indipendentemente dall'inchiostro rilevato.
        Utile per escludere la zona del banner data/giorno.

    Returns
    -------
    list of dict
        Come l'input ``boxes`` con tre campi aggiuntivi:
        - ``'ink_density'`` : float, densità di inchiostro nella cella
        - ``'checked'``     : bool, True se ink_density >= ink_threshold
        - ``'uncertain'``   : bool, True se uncertain_threshold <= ink_density < ink_threshold
    """
    if handwritten_only.ndim == 3:
        gray = cv2.cvtColor(handwritten_only, cv2.COLOR_BGR2GRAY)
    else:
        gray = handwritten_only.copy()

    ink_mask = (gray < 128).astype(np.float32)

    results = []
    for box in boxes:
        x, y, w, h = box["bbox"]
        cx, cy = x + w // 2, y + h // 2

        excluded = exclude_regions and any(
            x1 <= cx <= x2 and y1 <= cy <= y2
            for (x1, y1, x2, y2) in exclude_regions
        )

        # Margine adattivo: cap a 6 per riquadri grandi (min_side ≥ 36px).
        # A m=6 il bordo stampato (~5px) è completamente escluso; l'ink residuo
        # è solo del segno a mano, anche se vicino al bordo.
        m = min(max(margin, min(w, h) // 6), 6)
        img_h, img_w = ink_mask.shape
        x0 = max(x + m, 0);      x1 = min(x + w - m, img_w)
        y0 = max(y + m, 0);      y1 = min(y + h - m, img_h)
        if x1 <= x0 or y1 <= y0:
            density = 0.0
        else:
            density = float(ink_mask[y0:y1, x0:x1].mean())

        # Soglia adattiva: solo i box con m=6 (grandi, bordo completamente escluso)
        # usano la soglia bassa. Quelli con m=5 (min_side 30-35px) mantengono
        # soglia alta perché il bordo stampato può ancora "sanguinare" nell'area interna.
        effective_threshold = ink_threshold if m >= 6 else max(ink_threshold, 0.06)

        is_checked   = (not excluded) and (density >= effective_threshold)
        is_uncertain = (
            (not excluded)
            and (not is_checked)
            and (uncertain_threshold > 0)
            and (density >= uncertain_threshold)
        )
        entry = dict(box)
        entry["ink_density"] = round(density, 4)
        entry["checked"]     = is_checked
        entry["uncertain"]   = is_uncertain
        results.append(entry)

    return results


def draw_box_marks(
    image: np.ndarray,
    marked_boxes: List[Dict[str, Any]],
    *,
    color_checked: tuple = (0, 200, 0),
    color_uncertain: tuple = (220, 200, 0),
    color_empty: tuple = (180, 180, 180),
    color_ellipse: tuple = (0, 0, 255),
    thickness: int = 2,
    show_density: bool = False,
    show_empty: bool = False,
    show_ellipse: bool = True,
) -> np.ndarray:
    """
    Visualizza le celle del modulo: verde se segnate, giallo se incerte, grigio se vuote.

    Parameters
    ----------
    show_empty : bool
        Se False (default) disegna solo le celle segnate (verde) e incerte (giallo).
        Se True disegna anche le celle vuote in grigio — utile per debug.
    show_ellipse : bool
        Se True (default) disegna un'ellisse blu iscritta in ogni riquadro analizzato.

    Returns
    -------
    np.ndarray
        Immagine RGB annotata, pronta per ``plt.imshow``.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for b in marked_boxes:
        x, y, w, h = b["bbox"]
        is_uncertain = b.get("uncertain", False)

        if show_ellipse:
            cx, cy = x + w // 2, y + h // 2
            ax, ay = max(w // 2 - 2, 1), max(h // 2 - 2, 1)
            cv2.ellipse(out, (cx, cy), (ax, ay), 0, 0, 360, color_ellipse, 1)

        if not b["checked"] and not is_uncertain and not show_empty:
            continue

        if b["checked"]:
            color = color_checked
        elif is_uncertain:
            color = color_uncertain
        else:
            color = color_empty

        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if show_density:
            cv2.putText(out, f"{b['ink_density']:.2f}",
                        (x, max(y - 3, 10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    return out


def draw_crosses(
    image: np.ndarray,
    crosses: List[Dict[str, Any]],
    *,
    color: tuple = (255, 0, 0),
    thickness: int = 2,
    show_score: bool = True,
) -> np.ndarray:
    """
    Disegna i bounding box delle crocette sull'immagine.

    Parameters
    ----------
    image : np.ndarray
        Immagine sorgente (grayscale o BGR).
    crosses : list
        Output di :func:`detect_crosses`.
    color : tuple
        Colore RGB del rettangolo (default rosso).
    thickness : int
        Spessore del rettangolo in pixel.
    show_score : bool
        Se True stampa il punteggio sopra ogni rettangolo.

    Returns
    -------
    np.ndarray
        Copia dell'immagine in formato **RGB** con le annotazioni,
        pronta per ``plt.imshow``.
    """
    if image.ndim == 2:
        out = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    else:
        out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    for c in crosses:
        x, y, w, h = c["bbox"]
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
        if show_score:
            label = f"{c['score']:.2f}"
            cv2.putText(
                out, label,
                (x, max(y - 4, 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4, color, 1, cv2.LINE_AA,
            )
    return out
