"""
Image-plane spot scanning over a field x wavelength grid via the ZOS-API.

Answers: "scan the image plane" — specify a range of field angles (e.g.
-1..1 deg over 20 points in X, -0.5..0.5 deg over 30 points in Y) and a set
of wavelengths; OpticStudio traces a bundle of real rays at every grid
point in ONE batch ray trace, the rays are centroided per point, and the
resulting RMS spot-size / centroid maps can be interpolated anywhere in
the scanned field.

Why batch ray tracing (IBatchRayTrace) and not PSF analyses:
  * arbitrary (Hx, Hy) — no need to define 600 field points in the file
  * one tool run traces every ray for every grid point and wavelength
  * exactly the requested semantics: launch N rays, centroid, spot size

Field normalization: batch rays are launched at NORMALIZED field
coordinates, scaled by the maximum field defined in the prescription
(radially or rectangularly, per the system's normalization setting).  The
scan range must therefore lie within the defined maximum field — the scan
validates this and tells you how to fix it if not.

Caveat: ray spots are GEOMETRIC.  Near the diffraction limit the true
image blur is bounded below by the Airy core; see `with_diffraction_floor`.

Usage (with snr_calc.zemax_iface):

    from snr_calc.zemax_iface import ZOSConnection
    from snr_calc.zemax_field_scan import FieldScanConfig, scan_image_plane

    scan_cfg = FieldScanConfig(
        x_range=(-1 * u.deg, 1 * u.deg), n_x=20,
        y_range=(-0.5 * u.deg, 0.5 * u.deg), n_y=30,
        wavelengths=(486 * u.nm, 550 * u.nm),   # None -> all in file
        pupil_rings=6,              # 127 rays per field point
    )
    zos = ZOSConnection()
    try:
        zos.load(str(cfg.zemax_file))
        result = scan_image_plane(zos, scan_cfg)
    finally:
        zos.close()

    rms = result.rms_at(0.3 * u.deg, -0.1 * u.deg)      # interpolated, um
    fx, fy = result.worst_field()                        # largest spot
"""
from __future__ import annotations

from pathlib import Path
import sys
import warnings
if sys.platform != "win32":
    raise RuntimeError(
        "Zemax field scan only works on Windows with the ZOS-API")

import logging
from dataclasses import dataclass
from typing import Optional, Sequence

import astropy.units as u
import numpy as np
from tqdm import tqdm
from xarray import DataArray

from .zemax_iface import ZOSConnection
from .version import __version__

logger = logging.getLogger(__name__)


# ======================================================================
# Generic class: FieldScanConfig
# ======================================================================
@dataclass(frozen=True)
class FieldScanConfig:
    """Any image-plane scan: a field grid, wavelengths, and a ray bundle."""

    x_range: tuple[u.Quantity, u.Quantity] = (-1 * u.deg, 1 * u.deg)
    n_x: int = 20
    y_range: tuple[u.Quantity, u.Quantity] = (-0.5 * u.deg, 0.5 * u.deg)
    n_y: int = 30
    # Wavelengths to trace, as quantities (e.g. [550 * u.nm, 486 * u.nm]);
    # None -> all wavelengths defined in the file.  Requested values that
    # match a defined table entry (within 0.01 nm) reuse it; others are
    # temporarily added to the system's wavelength table for the trace
    # and removed again afterwards.
    wavelengths: Optional[tuple] = None
    # Hexapolar pupil sampling: `pupil_rings` rings -> 1 + 3 n (n+1) rays
    # per field point per wavelength (6 rings = 127 rays).
    pupil_rings: int = 6
    # Rays per batch-trace run (chunked to bound .NET memory).
    batch_chunk: int = 200_000
    # Keep the individual ray landing coordinates in the result (needed
    # for true-spot-shape detector rendering; small memory cost).
    keep_rays: bool = False
    # Field normalization to apply before scanning: "radial",
    # "rectangular", or None to leave the prescription as-is.  With
    # rectangular normalization the scan range only needs to fit the
    # per-axis maxima (|x| <= max|X|, |y| <= max|Y|) instead of the
    # corner radius, which usually matches a rectangular detector scan.
    normalization: Optional[str] = None

    # Optional explicit sample points (quantities), overriding
    # range/count; may be NON-UNIFORM, e.g. to densify regions where the
    # spot shape changes rapidly along a spectrograph slit.
    x_points: Optional[tuple] = None
    y_points: Optional[tuple] = None

    def x_axis(self, unit) -> np.ndarray:
        if self.x_points is not None:
            return np.sort([q.to(unit).value for q in self.x_points])
        lo, hi = (q.to(unit).value for q in self.x_range)
        return np.linspace(lo, hi, self.n_x)

    def y_axis(self, unit) -> np.ndarray:
        if self.y_points is not None:
            return np.sort([q.to(unit).value for q in self.y_points])
        lo, hi = (q.to(unit).value for q in self.y_range)
        return np.linspace(lo, hi, self.n_y)

    def scan_image_plane(self, zos: ZOSConnection) -> SpotScanResult:
        """Trace ray bundles over the field grid and reduce to spot maps."""
        ZOSAPI = zos.ZOSAPI
        sd = zos.system.SystemData

        # -- wavelengths -------------------------------------------------------
        added_waves: list[int] = []
        if self.wavelengths is None:
            n_wave_def = sd.Wavelengths.NumberOfWavelengths
            wave_indices = list(range(1, n_wave_def + 1))
            wavelengths_um = [float(sd.Wavelengths.GetWavelength(w).Wavelength)
                              for w in wave_indices]
        else:
            wave_indices, wavelengths_um, added_waves = _resolve_wavelengths(
                zos, self.wavelengths
            )

        # -- geometry ----------------------------------------------------------
        if self.normalization is not None:
            set_field_normalization(zos, self.normalization)
        field_unit, unit_name = _field_unit(zos)
        x_ax = self.x_axis(field_unit)
        y_ax = self.y_axis(field_unit)
        to_norm = _field_normalizer(zos, x_ax, y_ax)
        px, py = hexapolar_pupil(self.pupil_rings)
        rays_per_point = px.size

        lens_to_um = 1000.0  # lens units are mm for these systems
        image_surf = zos.system.LDE.NumberOfSurfaces - 1
        opd_none = getattr(ZOSAPI.Tools.RayTrace.OPDMode, "None")

        n_points = x_ax.size * y_ax.size * len(wave_indices)
        n_rays = n_points * rays_per_point
        logger.info(
            "scan: %d x %d field grid, %d wavelength(s), %d rays/point "
            "-> %d rays total",
            x_ax.size, y_ax.size, len(wave_indices), rays_per_point, n_rays,
        )

        # ray bookkeeping: rays are read back in the order they were added
        jobs = [(iw, iy, ix)
                for iw in range(len(wave_indices))
                for iy in range(y_ax.size)
                for ix in range(x_ax.size)]

        sum_x = np.zeros((len(wave_indices), y_ax.size, x_ax.size))
        sum_y = np.zeros_like(sum_x)
        sum_r2 = np.zeros_like(sum_x)
        n_valid = np.zeros_like(sum_x, dtype=int)
        xy_store: list[list[tuple[int, float, float]]] = \
            [[] for _ in range(y_ax.size * x_ax.size)]

        # -- chunked batch trace -------------------------------------------------
        chunk_pts = max(1, self.batch_chunk // rays_per_point)
        pbar = tqdm(
            total=n_rays, unit="rays", unit_scale=True,
            desc="Raycasting", dynamic_ncols=True,
        )
        try:
            for start in range(0, len(jobs), chunk_pts):
                chunk = jobs[start:start + chunk_pts]
                tool = zos.system.Tools.OpenBatchRayTrace()
                try:
                    norm_unpol = tool.CreateNormUnpol(
                        len(chunk) * rays_per_point,
                        ZOSAPI.Tools.RayTrace.RaysType.Real, image_surf,
                    )
                    for iw, iy, ix in chunk:
                        hx, hy = to_norm(x_ax[ix], y_ax[iy])
                        for k in range(rays_per_point):
                            norm_unpol.AddRay(
                                wave_indices[iw], hx, hy,
                                float(px[k]), float(py[k]), opd_none,
                            )
                    logger.debug(
                        "scan: tracing rays %d..%d of %d ...",
                        start * rays_per_point,
                        (start + len(chunk)) * rays_per_point, n_rays,
                    )
                    pbar.update(len(chunk) * rays_per_point)
                    tool.RunAndWaitForCompletion()
                    norm_unpol.StartReadingResults()

                    reader, first = _make_ray_reader(norm_unpol)
                    res, fresh = first, False
                    for j, (iw, iy, ix) in enumerate(chunk):
                        for _ in range(rays_per_point):
                            if fresh:
                                raw = reader()
                                res = (bool(raw[0]), int(raw[1]), int(raw[2]),
                                       int(raw[3]), float(raw[4]), float(raw[5]))
                            fresh = True
                            ok, _, err, vig, xr, yr = res
                            if not ok:
                                raise RuntimeError(
                                    "Batch ray trace returned fewer results than "
                                    "rays added."
                                )
                            if err == 0 and vig == 0:
                                x_um, y_um = xr * lens_to_um, yr * lens_to_um
                                sum_x[iw, iy, ix] += x_um
                                sum_y[iw, iy, ix] += y_um
                                n_valid[iw, iy, ix] += 1
                                xy_store[iy * x_ax.size + ix].append(
                                    (iw, x_um, y_um)
                                )
                finally:
                    tool.Close()

        finally:
            pbar.close()
            # restore the prescription: remove wavelengths this scan added
            for idx in sorted(added_waves, reverse=True):
                zos.system.SystemData.Wavelengths.RemoveWavelength(idx)
                logger.info("wavelengths: removed temporary table "
                            "entry %d", idx)

        # -- reduce: per-wavelength centroid + RMS ---------------------------------
        with np.errstate(invalid="ignore", divide="ignore"):
            cx = np.where(n_valid > 0, sum_x / n_valid, np.nan)
            cy = np.where(n_valid > 0, sum_y / n_valid, np.nan)
        rms = np.full_like(cx, np.nan)
        rms_poly = np.full((y_ax.size, x_ax.size), np.nan)
        for iy in range(y_ax.size):
            for ix in range(x_ax.size):
                pts = xy_store[iy * x_ax.size + ix]
                if not pts:
                    continue
                arr = np.asarray(pts)                      # (n, 3): iw, x, y
                for iw in range(len(wave_indices)):
                    sel = arr[arr[:, 0] == iw]
                    if sel.size:
                        dx = sel[:, 1] - cx[iw, iy, ix]
                        dy = sel[:, 2] - cy[iw, iy, ix]
                        rms[iw, iy, ix] = np.sqrt(np.mean(dx ** 2 + dy ** 2))
                # polychromatic: pool all rays around a common centroid
                dx = arr[:, 1] - arr[:, 1].mean()
                dy = arr[:, 2] - arr[:, 2].mean()
                rms_poly[iy, ix] = np.sqrt(np.mean(dx ** 2 + dy ** 2))

        n_dead = int(np.sum(n_valid.sum(axis=0) == 0))
        if n_dead:
            logger.warning("scan: %d grid point(s) had no valid rays "
                           "(vignetting or trace errors) -> NaN", n_dead)

        return SpotScanResult(
            x_field=x_ax, y_field=y_ax,
            field_unit=field_unit, unit_name=unit_name,
            wave_indices=wave_indices, wavelengths_um=wavelengths_um,
            centroid_x_um=cx, centroid_y_um=cy,
            rms_um=rms, rms_poly_um=rms_poly, n_valid=n_valid,
            rays=(_rays_ndarray(xy_store, x_ax, y_ax)
                  if self.keep_rays else None),
        )


def hexapolar_pupil(n_rings: int) -> tuple[np.ndarray, np.ndarray]:
    """Hexapolar (Px, Py) pupil pattern: center + rings of 6k points."""
    px, py = [0.0], [0.0]
    for k in range(1, n_rings + 1):
        r = k / n_rings
        th = np.linspace(0.0, 2.0 * np.pi, 6 * k, endpoint=False)
        px.extend(r * np.cos(th))
        py.extend(r * np.sin(th))
    return np.asarray(px), np.asarray(py)


# ======================================================================
# Scan result
# ======================================================================
@dataclass(frozen=True)
class SpotScanResult:
    """Gridded spot statistics; all lengths in um, all fields in degrees.

    Per-wavelength arrays are shaped (n_wave, n_y, n_x); the polychromatic
    arrays (all selected wavelengths pooled around a common centroid) are
    (n_y, n_x).  Field axes are degrees for angle-type prescriptions and
    mm for object/image-height (slit) prescriptions — see `unit_name`.  Grid points where no ray traced (vignetted / ray errors)
    are NaN.
    """

    x_field: np.ndarray               # (n_x,) scan axis [field_unit]
    y_field: np.ndarray               # (n_y,) scan axis [field_unit]
    field_unit: object                # astropy unit of the field axes
    unit_name: str                    # "deg" or "mm"
    wave_indices: list[int]
    wavelengths_um: list[float]
    centroid_x_um: np.ndarray         # (n_wave, n_y, n_x)
    centroid_y_um: np.ndarray
    rms_um: np.ndarray                # (n_wave, n_y, n_x) RMS spot radius
    rms_poly_um: np.ndarray           # (n_y, n_x) polychromatic
    n_valid: np.ndarray               # (n_wave, n_y, n_x) rays used
    # Optional per-ray data (keep_rays=True): a single (N, 5) float
    # ndarray, one row per traced ray, columns RAY_COLUMNS =
    # (wave_pos, field_x, field_y, x_um, y_um) — i.e. BOTH the casting
    # field coordinates (in field_unit: deg or mm) and the image-plane
    # landing position (um, image-surface local frame).
    rays: Optional[np.ndarray] = None

    RAY_COLUMNS = ("wave_pos", "field_x", "field_y", "x_um", "y_um")

    # ======================================================================
    # The scan
    # ======================================================================
    @staticmethod
    def concat_scans(results: Sequence[SpotScanResult]) -> SpotScanResult:
        """Combine multiple SpotScanResults into one.

        Two merge modes, chosen automatically:
        * FIELD merge — identical wavelengths and one identical field axis:
            results are concatenated along the other (slit) axis and sorted.
            Use case: densifying slit regions with separate scans (e.g. after
            an undersampling warning) without redoing the whole slit.
        * WAVELENGTH merge — identical field grids: results are stacked
            along the wavelength axis.  Use case: scanning lines in batches.

        Per-ray data merges too: rays carry absolute field coordinates, so
        no re-indexing of positions is needed (wavelength merge offsets the
        wave_pos column).  Duplicate field samples / repeated grids raise.
        """
        if len(results) < 2:
            return results[0]
        r0 = results[0]
        if any(r.unit_name != r0.unit_name for r in results):
            raise ValueError("cannot merge scans with different field units")
        same_lam = all(
            np.array_equal(r.wavelengths_um, r0.wavelengths_um)
            for r in results
        )
        same_x = all(np.array_equal(r.x_field, r0.x_field) for r in results)
        same_y = all(np.array_equal(r.y_field, r0.y_field) for r in results)

        def cat_rays(offsets=None):
            parts = []
            for i, r in enumerate(results):
                if r.rays is None:
                    return None
                rr = r.rays
                if offsets is not None and offsets[i]:
                    rr = rr.copy()
                    rr[:, 0] += offsets[i]
                parts.append(rr)
            return np.concatenate(parts)

        if same_lam and (same_x or same_y):
            # ---- field merge along the non-identical axis ----
            axis = "y" if same_x else "x"
            coords = np.concatenate([getattr(r, f"{axis}_field")
                                    for r in results])
            if np.unique(coords).size != coords.size:
                raise ValueError(f"duplicate {axis} field samples across "
                                 f"scans; cannot merge")
            order = np.argsort(coords)
            np_axis = 1 if axis == "y" else 2

            def cat(name):
                a = np.concatenate([getattr(r, name) for r in results],
                                   axis=np_axis)
                return np.take(a, order, axis=np_axis)
            rms_poly = np.take(
                np.concatenate([r.rms_poly_um for r in results],
                               axis=np_axis - 1),
                order, axis=np_axis - 1)
            kw = dict(x_field=r0.x_field, y_field=coords[order]) \
                if axis == "y" else \
                dict(x_field=coords[order], y_field=r0.y_field)
            return SpotScanResult(
                field_unit=r0.field_unit, unit_name=r0.unit_name,
                wave_indices=r0.wave_indices,
                wavelengths_um=r0.wavelengths_um,
                centroid_x_um=cat("centroid_x_um"),
                centroid_y_um=cat("centroid_y_um"),
                rms_um=cat("rms_um"), rms_poly_um=rms_poly,
                n_valid=cat("n_valid"), rays=cat_rays(), **kw)

        if same_x and same_y:
            # ---- wavelength merge on the identical grid ----
            lams, seen = [], set()
            for r in results:
                for lam in r.wavelengths_um:
                    if lam in seen:
                        raise ValueError(f"wavelength {lam} um appears in "
                                         f"multiple scans")
                    seen.add(lam)
                    lams.append(lam)
            offsets = np.cumsum([0] + [len(r.wavelengths_um)
                                       for r in results[:-1]])

            def cat0(name):
                return np.concatenate([getattr(r, name) for r in results],
                                      axis=0)
            # polychromatic RMS is NOT reconstructible from per-wave stats;
            # conservatively recompute from rays if available, else NaN
            rays = cat_rays(offsets)
            rms_poly = np.full_like(r0.rms_poly_um, np.nan)
            if rays is not None:
                for iy, yv in enumerate(r0.y_field):
                    for ix, xv in enumerate(r0.x_field):
                        sel = (np.isclose(rays[:, 1], xv)
                               & np.isclose(rays[:, 2], yv))
                        if sel.any():
                            xy = rays[sel][:, 3:5]
                            rms_poly[iy, ix] = np.sqrt(
                                np.mean(np.sum((xy - xy.mean(0)) ** 2, 1)))
            return SpotScanResult(
                x_field=r0.x_field, y_field=r0.y_field,
                field_unit=r0.field_unit, unit_name=r0.unit_name,
                wave_indices=[w for r in results for w in r.wave_indices],
                wavelengths_um=lams,
                centroid_x_um=cat0("centroid_x_um"),
                centroid_y_um=cat0("centroid_y_um"),
                rms_um=cat0("rms_um"), rms_poly_um=rms_poly,
                n_valid=cat0("n_valid"), rays=rays)

        raise ValueError(
            "scans are not mergeable: need identical wavelengths plus one "
            "shared field axis (field merge), or identical field grids "
            "(wavelength merge)")

    # ---- composition ----------------------------------------------------

    def __add__(self, other):
        """`a + b` concatenates scans in memory (see concat_scans)."""
        if not isinstance(other, SpotScanResult):
            return NotImplemented
        return self.concat_scans([self, other])

    def __radd__(self, other):
        """Supports sum(results): 0 + result -> result."""
        if other == 0:
            return self
        return NotImplemented

    # ---- persistence ------------------------------------------------------
    def save(self, path: Path, allow_pickle: bool = False) -> None:
        """Save to a compressed .npz (plain arrays only, no pickling).

        Round-trips through SpotScanResult.load(); loaded results can be
        concatenated in memory with `+` / concat_scans like fresh scans.
        """
        payload = dict(
            x_field=self.x_field, y_field=self.y_field,
            unit_name=np.asarray(self.unit_name),
            wave_indices=np.asarray(self.wave_indices, dtype=int),
            wavelengths_um=np.asarray(self.wavelengths_um, dtype=float),
            centroid_x_um=self.centroid_x_um,
            centroid_y_um=self.centroid_y_um,
            rms_um=self.rms_um, rms_poly_um=self.rms_poly_um,
            n_valid=self.n_valid,
            version=np.array(__version__),
        )
        if self.rays is not None:
            payload["rays"] = self.rays
        np.savez_compressed(path, allow_pickle=allow_pickle, **payload)

    @classmethod
    def load(cls, path: Path, allow_pickle: bool = False) -> "SpotScanResult":
        """Load a result saved with save()."""
        with np.load(path, allow_pickle=allow_pickle) as z:
            if (version := z.get("version")) != np.array(__version__):
                warnings.warn(
                    f"SpotScanResult version mismatch: file has {version}, "
                    f"current version is {__version__}"
                )
            unit_name = str(z["unit_name"])
            unit = {"deg": u.deg, "mm": u.mm}[unit_name]
            return cls(
                x_field=z["x_field"], y_field=z["y_field"],
                field_unit=unit, unit_name=unit_name,
                wave_indices=[int(w) for w in z["wave_indices"]],
                wavelengths_um=[float(w) for w in z["wavelengths_um"]],
                centroid_x_um=z["centroid_x_um"],
                centroid_y_um=z["centroid_y_um"],
                rms_um=z["rms_um"], rms_poly_um=z["rms_poly_um"],
                n_valid=z["n_valid"],
                rays=z["rays"] if "rays" in z.files else None,
            )

    def to_image(
        self, pixel_pitch: u.Quantity,
        detector_mm: Optional[tuple[float, float]] = None,
        method: str = "auto",
        detector_center_mm: tuple[float, float] = (0.0, 0.0),
        oversample: int = 4
    ) -> "DataArray":
        """Compute the detector image (no plotting).

        Coordinate convention: all positions are in the Zemax image surface's
        LOCAL frame, origin at the surface vertex (the batch-ray X/Y, i.e.
        the Footprint Diagram convention — not the chief-ray-relative Spot
        Diagram convention).  `detector_mm` places the pixel grid centered on
        `detector_center_mm` (default: the vertex); use the offset when the
        physical detector is mounted off the image-surface vertex.

        Returns (image, x_centers_mm, y_centers_mm, attrs); image is
        (n_pix_y, n_pix_x), row 0 at min y.

        method:
        "rays"     true spot shapes: traced ray landing coordinates are
                    deposited onto the pixels; between scanned slit samples
                    the two neighboring ray clouds are re-centered onto the
                    interpolated centroid and cross-faded, preserving coma
                    tails, astigmatic elongation, and other real structure.
                    Requires a scan with keep_rays=True.
        "gaussian" analytic approximation: sigma = RMS/sqrt(2), integrated
                    over pixel boundaries (per-axis erf).
        "auto"     "rays" when ray data is present, else "gaussian".

        oversample (rays method only): rays are deposited on a grid
        `oversample` times finer, then block-summed to the native pitch.
        This integrates flux over the pixel AREA and removes pixel-phase
        aliasing when the line width is below the pixel pitch (banding along
        a sharp line).  Flux is exactly conserved.  1 disables.
        """
        if method == "auto":
            method = "rays" if self.rays is not None else "gaussian"
        if method == "rays" and self.rays is None:
            raise ValueError("method='rays' needs a scan with keep_rays=True")
        if method not in ("rays", "gaussian"):
            raise ValueError(f"unknown method {method!r}")

        pitch_mm = pixel_pitch.to(u.um).value / 1000.0
        n_lines = len(self.wavelengths_um)
        n_x = self.x_field.size

        # per-line data: one sub-curve per column ACROSS the slit (the slit
        # runs along the larger scan axis).  A 2-D scan — e.g. several field
        # columns sampling a finite slit WIDTH — is rendered as parallel
        # sub-curves sharing the line's unit flux, not as one raveled
        # zigzag path.
        n_y = self.y_field.size
        slit_along_y = n_y >= n_x
        n_cols = n_x if slit_along_y else n_y
        if n_cols > 1:
            logger.info("render: %d columns across the slit per line",
                        n_cols)
        lines = []
        for k in range(n_lines):
            CX = self.centroid_x_um[k] / 1000.0        # (n_y, n_x) [mm]
            CY = self.centroid_y_um[k] / 1000.0
            W = self.rms_um[k] / 1000.0
            curves = []
            for c in range(n_cols):
                if slit_along_y:
                    cx, cy, w = CX[:, c], CY[:, c], W[:, c]
                    flats = np.arange(n_y) * n_x + c
                else:
                    cx, cy, w = CX[c, :], CY[c, :], W[c, :]
                    flats = c * n_x + np.arange(n_x)
                good = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(w)
                if not good.any():
                    continue
                cx, cy, w = cx[good], cy[good], w[good]
                clouds = None
                if method == "rays":
                    clouds = []
                    for flat in flats[good]:
                        iy, ix = divmod(int(flat), n_x)
                        r = self.rays_at(k, iy, ix) / 1000.0      # mm
                        clouds.append(r - [cx[len(clouds)],
                                           cy[len(clouds)]])     # centered
                curves.append((cx, cy, w, clouds))
            lines.append(curves if curves else None)

        # pixel-grid extent
        if detector_mm is not None:
            dw, dh = detector_mm
            cx0, cy0 = detector_center_mm
            x_min, x_max = cx0 - dw / 2, cx0 + dw / 2
            y_min, y_max = cy0 - dh / 2, cy0 + dh / 2
        else:
            curves_all = [C for L in lines if L is not None for C in L]
            allx = np.concatenate([C[0] for C in curves_all])
            ally = np.concatenate([C[1] for C in curves_all])
            if method == "rays":
                dev = max(
                    float(np.abs(cl).max()) for C in curves_all
                    for cl in C[3]
                )
            else:
                dev = 5.0 * max(float(C[2].max()) for C in curves_all)
            m = dev + 2.0 * pitch_mm
            x_min, x_max = allx.min() - m, allx.max() + m
            y_min, y_max = ally.min() - m, ally.max() + m
        n_px = int(np.ceil((x_max - x_min) / pitch_mm))
        n_py = int(np.ceil((y_max - y_min) / pitch_mm))
        x_edges = x_min + np.arange(n_px + 1) * pitch_mm
        y_edges = y_min + np.arange(n_py + 1) * pitch_mm
        ov = max(1, int(oversample)) if method == "rays" else 1
        pitch_f = pitch_mm / ov               # deposition grid (rays)
        n_pxf, n_pyf = n_px * ov, n_py * ov
        image = np.zeros((n_pyf, n_pxf))
        logger.info("render: %d x %d pixels at %g um pitch, method=%s, "
                    "oversample=%d", n_px, n_py,
                    pixel_pitch.to(u.um).value, method, ov)

        for k, L in enumerate(lines):
            if L is None:
                continue
            # diagnostics on the central column only (columns are near-copies)
            mid = L[len(L) // 2]
            if method == "rays" and mid[0].size > 1:
                _warn_undersampled_shape(self, k, mid[0], mid[1], mid[2])
            # each sub-curve carries an equal share of the line's unit flux
            for cx, cy, w, clouds in L:
                # fine samples along the line (arc length, ~pitch/2 steps)
                if cx.size > 1:
                    ds = np.hypot(np.diff(cx), np.diff(cy))
                    t = np.concatenate(([0.0], np.cumsum(ds)))
                    n_fine = max(cx.size,
                                 int(np.ceil(t[-1] / (pitch_f / 2.0))) + 1)
                    tf = np.linspace(0.0, t[-1], n_fine)
                    cxf = np.interp(tf, t, cx)
                    cyf = np.interp(tf, t, cy)
                    wf = np.interp(tf, t, w)
                    seg = np.clip(np.searchsorted(t, tf, side="right") - 1,
                                  0, cx.size - 2)
                    frac = np.where(np.diff(t)[seg] > 0,
                                    (tf - t[seg]) / np.diff(t)[seg], 0.0)
                else:
                    cxf, cyf, wf = cx, cy, w
                    seg = np.zeros(1, dtype=int)
                    frac = np.zeros(1)
                flux = 1.0 / (cxf.size * len(L))

                if method == "gaussian":
                    from scipy.special import erf

                    for j in range(cxf.size):
                        sigma = max(wf[j] / np.sqrt(2.0), 1e-9)
                        half = 4.0 * sigma
                        i0 = max(0, np.searchsorted(
                            x_edges, cxf[j] - half) - 1)
                        i1 = min(n_px, np.searchsorted(
                            x_edges, cxf[j] + half) + 1)
                        j0 = max(0, np.searchsorted(
                            y_edges, cyf[j] - half) - 1)
                        j1 = min(n_py, np.searchsorted(
                            y_edges, cyf[j] + half) + 1)
                        if i0 >= i1 or j0 >= j1:
                            continue
                        sq = sigma * np.sqrt(2.0)
                        fx = 0.5 * \
                            np.diff(erf((x_edges[i0:i1 + 1] - cxf[j]) / sq))
                        fy = 0.5 * \
                            np.diff(erf((y_edges[j0:j1 + 1] - cyf[j]) / sq))
                        image[j0:j1, i0:i1] += flux * np.outer(fy, fx)
                else:
                    def deposit(pts_x, pts_y, weight):
                        ip = np.floor((pts_x - x_min) / pitch_f).astype(int)
                        jp = np.floor((pts_y - y_min) / pitch_f).astype(int)
                        ok = (ip >= 0) & (ip < n_pxf) & (
                            jp >= 0) & (jp < n_pyf)
                        np.add.at(image, (jp[ok], ip[ok]), weight)

                    for j in range(cxf.size):
                        a, f = int(seg[j]), float(frac[j])
                        for cloud, wgt in ((clouds[a], (1.0 - f)),
                                           (clouds[min(a + 1, len(clouds) - 1)],
                                            f)):
                            if wgt <= 0.0 or cloud.shape[0] == 0:
                                continue
                            deposit(cloud[:, 0] + cxf[j], cloud[:, 1] + cyf[j],
                                    flux * wgt / cloud.shape[0])

        if ov > 1:
            image = image.reshape(n_py, ov, n_px, ov).sum(axis=(1, 3))

        x_centers = x_min + (np.arange(n_px) + 0.5) * pitch_mm
        y_centers = y_min + (np.arange(n_py) + 0.5) * pitch_mm
        attrs = {
            "units": "fraction of line flux per pixel",
            "pixel_pitch_um": float(pixel_pitch.to(u.um).value),
            "method": method,
            "wavelengths_nm": [lam * 1000.0 for lam in self.wavelengths_um],
            "line_x_mm": [float(np.mean(np.concatenate([C[0] for C in L])))
                          if L is not None else np.nan for L in lines],
            "line_y_mm": [float(np.mean(np.concatenate([C[1] for C in L])))
                          if L is not None else np.nan for L in lines],
            "detector_center_mm": list(detector_center_mm),
            "origin": "Zemax image-surface vertex (local frame)",
        }
        coords = {"x": x_centers, "y": y_centers}
        coords.update(_image_secondary_coords(
            self, x_centers, y_centers, attrs))
        return DataArray(image, dims=("y", "x"), coords=coords,
                         attrs=attrs, name="flux")

    def rays_at(self, wave_pos: int, iy: int, ix: int) -> np.ndarray:
        """(n, 2) ray landing coordinates [um] for one grid point/line.

        Selection is by the stored casting-field coordinates (not array
        indices), so it remains valid after concat_scans() reorders or
        merges grids.
        """
        if self.rays is None:
            raise ValueError("scan was run without keep_rays=True")
        r = self.rays
        sel = ((r[:, 0] == wave_pos)
               & np.isclose(r[:, 1], self.x_field[ix])
               & np.isclose(r[:, 2], self.y_field[iy]))
        return r[sel][:, 3:5]

    # ---- interpolation ---------------------------------------------------
    def _wave_pos(self, wavelength: u.Quantity, tol_um: float = 1e-3) -> int:
        """Array position of the stored wavelength nearest `wavelength`."""
        lam = wavelength.to(u.um).value
        pos = int(np.argmin(np.abs(np.asarray(self.wavelengths_um) - lam)))
        if abs(self.wavelengths_um[pos] - lam) > tol_um:
            raise ValueError(
                f"{lam:g} um was not scanned; available: "
                f"{self.wavelengths_um} um."
            )
        return pos

    def rms_interpolator(self, wavelength: Optional[u.Quantity] = None,
                         method: str = "linear"):
        """Interpolator f((y_field, x_field)) -> RMS spot radius [um].

        wavelength=None -> polychromatic map; otherwise the scanned
        wavelength nearest the given quantity (within 1 nm).  Uses
        scipy.interpolate.RegularGridInterpolator over the scan grid.
        """
        from scipy.interpolate import RegularGridInterpolator

        data = (self.rms_poly_um if wavelength is None
                else self.rms_um[self._wave_pos(wavelength)])
        # data is (n_y, n_x); grid axes ordered to match
        return RegularGridInterpolator(
            (self.y_field, self.x_field), data,
            method=method, bounds_error=True,
        )

    def rms_at(self, x: u.Quantity, y: u.Quantity,
               wavelength: Optional[u.Quantity] = None) -> float:
        """Interpolated RMS spot radius [um] at field (x, y)."""
        f = self.rms_interpolator(wavelength)
        return float(f((y.to(self.field_unit).value,
                        x.to(self.field_unit).value)))

    def worst_field(self) -> tuple[u.Quantity, u.Quantity]:
        """Field coordinates of the largest polychromatic RMS spot."""
        iy, ix = np.unravel_index(np.nanargmax(self.rms_poly_um),
                                  self.rms_poly_um.shape)
        return (self.x_field[ix] * self.field_unit,
                self.y_field[iy] * self.field_unit)

    def with_diffraction_floor(self, wfno: float,
                               wavelength: u.Quantity) -> np.ndarray:
        """Polychromatic RMS combined in quadrature with the Airy core.

        Geometric ray spots underestimate the blur near the diffraction
        limit; sigma_diff ~ 0.42 * lambda * F# approximates the Airy-core
        RMS radius.  Returns a (n_y, n_x) array [um].
        """
        sigma_diff = 0.42 * wavelength.to(u.um).value * wfno
        return np.sqrt(self.rms_poly_um ** 2 + sigma_diff ** 2)


def set_field_normalization(zos, mode: str) -> str:
    """Set the prescription's field normalization from Python.

    mode: "radial" or "rectangular" (case-insensitive).  Sets
    SystemData.Fields.Normalization to the corresponding
    ZOSAPI.SystemData.FieldNormalizationType member, reads it back to
    verify it applied, and returns the applied value as a string.

    Note this changes the loaded system's state (like editing the System
    Explorer in the GUI): every subsequent normalized-field computation —
    batch rays, merit operands with Hx/Hy — is affected until it is set
    back or the file is reloaded.
    """
    members = {"radial": "Radial", "rectangular": "Rectangular"}
    try:
        member = members[mode.lower()]
    except KeyError:
        raise ValueError(
            f"Unknown normalization mode {mode!r}; use "
            f"{sorted(members)}."
        ) from None
    fd = zos.system.SystemData.Fields
    before = str(fd.Normalization)
    fd.Normalization = getattr(
        zos.ZOSAPI.SystemData.FieldNormalizationType, member
    )
    applied = str(fd.Normalization)
    logger.info("normalization: %s -> %s (requested %s)",
                before, applied, member)
    if member.lower() not in applied.lower():
        raise RuntimeError(
            f"Field normalization did not apply: requested {member}, "
            f"system reports {applied!r}."
        )
    return applied


# ======================================================================
# Field normalization
# ======================================================================
def _field_unit(zos):
    """Field unit implied by the prescription's field type.

    Angle fields scan in degrees; object/image-height fields (the usual
    choice for a spectrograph slit) scan in lens units, assumed mm.
    Returns (astropy_unit, name).
    """
    ftype = str(zos.system.SystemData.Fields.GetFieldType())
    if "Angle" in ftype:
        return u.deg, "deg"
    return u.mm, "mm"


def _field_normalizer(zos, x_ax: np.ndarray, y_ax: np.ndarray):
    """Map scan coordinates (field units) -> normalized (Hx, Hy).

    Batch rays use normalized fields, scaled by the prescription's maximum
    defined field, either radially (|H| <= 1 on the radius of the largest
    field) or rectangularly (Hx, Hy each scaled by the max |X|, |Y|).
    Raises if the requested scan exceeds the defined maximum field.
    """
    fd = zos.system.SystemData.Fields
    xs, ys = [], []
    for i in range(1, fd.NumberOfFields + 1):
        f = fd.GetField(i)
        xs.append(float(f.X))
        ys.append(float(f.Y))
    xs, ys = np.asarray(xs), np.asarray(ys)

    norm = str(getattr(fd, "Normalization", "Radial"))
    if "Rectangular" in norm:
        fx_max = float(np.max(np.abs(xs)))
        fy_max = float(np.max(np.abs(ys)))
        if (np.max(np.abs(x_ax)) > fx_max + 1e-12
                or np.max(np.abs(y_ax)) > fy_max + 1e-12):
            raise ValueError(
                f"Scan range (|x| <= {np.max(np.abs(x_ax)):g}, "
                f"|y| <= {np.max(np.abs(y_ax)):g}) exceeds the "
                f"defined maximum field ({fx_max:g}, {fy_max:g}). "
                f"Add/extend a field point in the prescription to cover "
                f"the scan corner."
            )
        logger.debug("scan: rectangular normalization, fmax=(%g, %g)",
                     fx_max, fy_max)
        return lambda x, y: (x / fx_max if fx_max else 0.0,
                             y / fy_max if fy_max else 0.0)

    r_max = float(np.max(np.hypot(xs, ys)))
    r_need = float(np.max(np.hypot(*np.meshgrid(x_ax, y_ax))))
    if r_need > r_max + 1e-12:
        raise ValueError(
            f"Scan corner radius {r_need:g} exceeds the defined "
            f"maximum radial field {r_max:g}. Add/extend a field "
            f"point in the prescription (e.g. at the scan corner)."
        )
    logger.debug("scan: radial normalization, rmax=%g", r_max)
    return lambda x, y: (x / r_max, y / r_max)


# ======================================================================
# Batch ray reading (pythonnet 2.x / 3.x tolerant)
# ======================================================================
def _make_ray_reader(norm_unpol):
    """Return a callable yielding
    (ok, raynum, err, vig, x_lens_units, y_lens_units) per ray.

    ReadNextResult has 14 .NET `out` parameters; depending on the
    pythonnet version it is called with no arguments or with placeholder
    values, and the outputs come back as a returned tuple either way.
    """
    def read_noargs():
        return norm_unpol.ReadNextResult()

    def read_dummies():
        return norm_unpol.ReadNextResult(
            0, 0, 0,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
        )

    for reader in (read_noargs, read_dummies):
        try:
            probe = reader()
        except TypeError:
            continue
        # (ok, rayNumber, errCode, vigCode, X, Y, Z, L, M, N, ...)

        def unpack(res):
            return bool(res[0]), int(res[1]), int(res[2]), int(res[3]), \
                float(res[4]), float(res[5])
        return reader, unpack(probe)
    raise RuntimeError(
        "Could not call IRayTraceNormUnpolData.ReadNextResult with either "
        "pythonnet calling convention."
    )


def _resolve_wavelengths(zos, wavelengths,
                         tol_um: float = 1e-5
                         ) -> tuple[list[int], list[float], list[int]]:
    """Map requested wavelength quantities to system wavelength numbers.

    Reuses table entries matching within `tol_um` (0.01 nm); otherwise
    appends a new entry via AddWavelength.  Returns
    (wave_numbers, values_um, added_numbers) — the caller must remove
    `added_numbers` (in descending order) after tracing to restore the
    prescription.
    """
    wl_table = zos.system.SystemData.Wavelengths
    defined = [float(wl_table.GetWavelength(i).Wavelength)
               for i in range(1, wl_table.NumberOfWavelengths + 1)]
    indices, values_um, added = [], [], []
    for q in wavelengths:
        lam = q.to(u.um).value
        diffs = [abs(d - lam) for d in defined]
        if diffs and min(diffs) <= tol_um:
            idx = int(np.argmin(diffs)) + 1
            logger.debug("wavelengths: %g um reuses table entry %d",
                         lam, idx)
        else:
            wl_table.AddWavelength(lam, 1.0)
            idx = wl_table.NumberOfWavelengths
            defined.append(lam)
            added.append(idx)
            logger.info("wavelengths: added %g um to the table as "
                        "entry %d (will be removed after the scan)",
                        lam, idx)
        indices.append(idx)
        values_um.append(lam)
    return indices, values_um, added


def _rays_ndarray(xy_store, x_ax: np.ndarray, y_ax: np.ndarray) -> np.ndarray:
    """Flatten per-grid-point ray lists into one (N, 5) array carrying
    both the casting field coordinates and the landing positions."""
    chunks = []
    for flat, pts in enumerate(xy_store):
        if not pts:
            continue
        a = np.asarray(pts, dtype=float).reshape(-1, 3)  # (iw, x, y)
        iy, ix = divmod(flat, x_ax.size)
        out = np.empty((a.shape[0], 5))
        out[:, 0] = a[:, 0]
        out[:, 1] = x_ax[ix]
        out[:, 2] = y_ax[iy]
        out[:, 3:5] = a[:, 1:3]
        chunks.append(out)
    return (np.concatenate(chunks) if chunks
            else np.empty((0, 5)))


# ======================================================================
# Spectrograph: spectral lines on the image plane
# ======================================================================
def scan_spectral_lines(zos, wavelengths, slit_range=None,
                        n_slit: int = 41,
                        slit_points=None,
                        slit_axis: str = "y",
                        cross_field: Optional[u.Quantity] = None,
                        pupil_rings: int = 4,
                        normalization: Optional[str] = None,
                        keep_rays: bool = True
                        ) -> SpotScanResult:
    """Scan spectral lines: 1-D field sweep along the slit x wavelengths.

    wavelengths: spectral lines as quantities (e.g. (486.1*u.nm, 587.6*u.nm))
    slit_range:  (lo, hi) field extent along the slit — angle quantities
                 for angle-field prescriptions, lengths (mm) for
                 object-height (slit) prescriptions
    slit_axis:   "y" (default) or "x" — which field axis the slit lies on
    cross_field: fixed field value on the other axis (default 0)

    Each spectral line on the detector is then the centroid curve
    (centroid_x_um, centroid_y_um)[k] along the slit, with per-point width
    rms_um[k]; plot with `plot_spectral_lines`.
    """
    if slit_range is None and slit_points is None:
        raise ValueError("provide slit_range or slit_points")
    if slit_points is not None:
        ref = slit_points[0]
        pts = tuple(slit_points)
    else:
        assert slit_range is not None
        ref = slit_range[0]
        pts = None
    zero = 0 * (ref.unit if hasattr(ref, "unit") else u.deg)
    cross = cross_field if cross_field is not None else zero
    rng = slit_range if slit_range is not None else (ref, ref)
    if slit_axis == "y":
        cfg = FieldScanConfig(
            x_range=(cross, cross), n_x=1,
            y_range=rng, n_y=n_slit, y_points=pts,
            wavelengths=tuple(wavelengths),
            pupil_rings=pupil_rings,
            normalization=normalization,
            keep_rays=keep_rays
        )
    elif slit_axis == "x":
        cfg = FieldScanConfig(
            x_range=rng, n_x=n_slit, x_points=pts,
            y_range=(cross, cross), n_y=1,
            wavelengths=tuple(wavelengths),
            pupil_rings=pupil_rings,
            normalization=normalization,
            keep_rays=keep_rays
        )
    else:
        raise ValueError(f"slit_axis must be 'x' or 'y', got {slit_axis!r}")
    return cfg.scan_image_plane(zos)


def _wavelength_rgb(lam_um: float) -> tuple[float, float, float]:
    """Approximate display color for a wavelength (gray outside 380-750nm)."""
    w = lam_um * 1000.0
    if 380 <= w < 440:
        r, g, b = -(w - 440) / 60.0, 0.0, 1.0
    elif 440 <= w < 490:
        r, g, b = 0.0, (w - 440) / 50.0, 1.0
    elif 490 <= w < 510:
        r, g, b = 0.0, 1.0, -(w - 510) / 20.0
    elif 510 <= w < 580:
        r, g, b = (w - 510) / 70.0, 1.0, 0.0
    elif 580 <= w < 645:
        r, g, b = 1.0, -(w - 645) / 65.0, 0.0
    elif 645 <= w <= 750:
        r, g, b = 1.0, 0.0, 0.0
    else:
        return 0.5, 0.5, 0.5
    return r, g, b


def plot_spectral_lines(res: SpotScanResult, savepath: Optional[str] = None,
                        detector_mm: Optional[tuple[float, float]] = None,
                        ax=None):
    """Plot the spectral lines as they land on the image plane.

    Each scanned wavelength is drawn as its centroid curve along the slit
    (in detector mm), with a shaded band of +/- the RMS spot radius across
    the dispersion direction — so line curvature (smile), tilt, and blur
    are all visible.  `detector_mm=(width, height)` overlays the detector
    outline centered on the optical axis (Sony IMX267: (14.19, 10.38)).
    """
    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    else:
        fig = ax.figure

    # dispersion direction: the centroid coordinate that moves most
    # between wavelengths
    mx = [np.nanmean(res.centroid_x_um[k]) for k in range(len(
        res.wavelengths_um))]
    my = [np.nanmean(res.centroid_y_um[k]) for k in range(len(
        res.wavelengths_um))]
    disp_is_x = (np.ptp(mx) >= np.ptp(my))

    for k, lam in enumerate(res.wavelengths_um):
        x = res.centroid_x_um[k].ravel() / 1000.0     # mm
        y = res.centroid_y_um[k].ravel() / 1000.0
        w = res.rms_um[k].ravel() / 1000.0
        good = np.isfinite(x) & np.isfinite(y)
        x, y, w = x[good], y[good], w[good]
        if x.size == 0:
            continue
        color = _wavelength_rgb(lam)
        # order points along the slit direction for a clean curve
        order = np.argsort(y if disp_is_x else x)
        x, y, w = x[order], y[order], w[order]
        if disp_is_x:
            ax.fill_betweenx(y, x - w, x + w, color=color, alpha=0.3,
                             linewidth=0)
        else:
            ax.fill_between(x, y - w, y + w, color=color, alpha=0.3,
                            linewidth=0)
        ax.plot(x, y, "-", color=color, lw=1.5,
                label=f"{lam * 1000.0:.1f} nm")

    if detector_mm is not None:
        from matplotlib.patches import Rectangle
        dw, dh = detector_mm
        ax.add_patch(Rectangle((-dw / 2, -dh / 2), dw, dh, fill=False,
                               edgecolor="k", ls="--", lw=1.0,
                               label="detector"))
    ax.set_xlabel("image plane X [mm]")
    ax.set_ylabel("image plane Y [mm]")
    ax.set_title("Spectral lines on the image plane "
                 "(band = +/- RMS spot radius)")
    ax.set_aspect("equal")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"spectral-line plot saved to {savepath}")
    return fig, ax


def _render_image(res: SpotScanResult, pixel_pitch: u.Quantity,
                  detector_mm: Optional[tuple[float, float]] = None,
                  method: str = "auto",
                  detector_center_mm: tuple[float, float] = (0.0, 0.0),
                  oversample: int = 4):
    """Compute the detector image (no plotting).

    Coordinate convention: all positions are in the Zemax image surface's
    LOCAL frame, origin at the surface vertex (the batch-ray X/Y, i.e.
    the Footprint Diagram convention — not the chief-ray-relative Spot
    Diagram convention).  `detector_mm` places the pixel grid centered on
    `detector_center_mm` (default: the vertex); use the offset when the
    physical detector is mounted off the image-surface vertex.

    Returns (image, x_centers_mm, y_centers_mm, attrs); image is
    (n_pix_y, n_pix_x), row 0 at min y.

    method:
      "rays"     true spot shapes: traced ray landing coordinates are
                 deposited onto the pixels; between scanned slit samples
                 the two neighboring ray clouds are re-centered onto the
                 interpolated centroid and cross-faded, preserving coma
                 tails, astigmatic elongation, and other real structure.
                 Requires a scan with keep_rays=True.
      "gaussian" analytic approximation: sigma = RMS/sqrt(2), integrated
                 over pixel boundaries (per-axis erf).
      "auto"     "rays" when ray data is present, else "gaussian".

    oversample (rays method only): rays are deposited on a grid
    `oversample` times finer, then block-summed to the native pitch.
    This integrates flux over the pixel AREA and removes pixel-phase
    aliasing when the line width is below the pixel pitch (banding along
    a sharp line).  Flux is exactly conserved.  1 disables.
    """
    if method == "auto":
        method = "rays" if res.rays is not None else "gaussian"
    if method == "rays" and res.rays is None:
        raise ValueError("method='rays' needs a scan with keep_rays=True")
    if method not in ("rays", "gaussian"):
        raise ValueError(f"unknown method {method!r}")

    pitch_mm = pixel_pitch.to(u.um).value / 1000.0
    n_lines = len(res.wavelengths_um)
    n_x = res.x_field.size

    # per-line data: one sub-curve per column ACROSS the slit (the slit
    # runs along the larger scan axis).  A 2-D scan — e.g. several field
    # columns sampling a finite slit WIDTH — is rendered as parallel
    # sub-curves sharing the line's unit flux, not as one raveled
    # zigzag path.
    n_y = res.y_field.size
    slit_along_y = n_y >= n_x
    n_cols = n_x if slit_along_y else n_y
    if n_cols > 1:
        logger.info("render: %d columns across the slit per line",
                    n_cols)
    lines = []
    for k in range(n_lines):
        CX = res.centroid_x_um[k] / 1000.0        # (n_y, n_x) [mm]
        CY = res.centroid_y_um[k] / 1000.0
        W = res.rms_um[k] / 1000.0
        curves = []
        for c in range(n_cols):
            if slit_along_y:
                cx, cy, w = CX[:, c], CY[:, c], W[:, c]
                flats = np.arange(n_y) * n_x + c
            else:
                cx, cy, w = CX[c, :], CY[c, :], W[c, :]
                flats = c * n_x + np.arange(n_x)
            good = np.isfinite(cx) & np.isfinite(cy) & np.isfinite(w)
            if not good.any():
                continue
            cx, cy, w = cx[good], cy[good], w[good]
            clouds = None
            if method == "rays":
                clouds = []
                for flat in flats[good]:
                    iy, ix = divmod(int(flat), n_x)
                    r = res.rays_at(k, iy, ix) / 1000.0      # mm
                    clouds.append(r - [cx[len(clouds)],
                                       cy[len(clouds)]])     # centered
            curves.append((cx, cy, w, clouds))
        lines.append(curves if curves else None)

    # pixel-grid extent
    if detector_mm is not None:
        dw, dh = detector_mm
        cx0, cy0 = detector_center_mm
        x_min, x_max = cx0 - dw / 2, cx0 + dw / 2
        y_min, y_max = cy0 - dh / 2, cy0 + dh / 2
    else:
        curves_all = [C for L in lines if L is not None for C in L]
        allx = np.concatenate([C[0] for C in curves_all])
        ally = np.concatenate([C[1] for C in curves_all])
        if method == "rays":
            dev = max(float(np.abs(cl).max()) for C in curves_all
                      for cl in C[3])
        else:
            dev = 5.0 * max(float(C[2].max()) for C in curves_all)
        m = dev + 2.0 * pitch_mm
        x_min, x_max = allx.min() - m, allx.max() + m
        y_min, y_max = ally.min() - m, ally.max() + m
    n_px = int(np.ceil((x_max - x_min) / pitch_mm))
    n_py = int(np.ceil((y_max - y_min) / pitch_mm))
    x_edges = x_min + np.arange(n_px + 1) * pitch_mm
    y_edges = y_min + np.arange(n_py + 1) * pitch_mm
    ov = max(1, int(oversample)) if method == "rays" else 1
    pitch_f = pitch_mm / ov               # deposition grid (rays)
    n_pxf, n_pyf = n_px * ov, n_py * ov
    image = np.zeros((n_pyf, n_pxf))
    logger.info("render: %d x %d pixels at %g um pitch, method=%s, "
                "oversample=%d", n_px, n_py,
                pixel_pitch.to(u.um).value, method, ov)

    for k, L in enumerate(lines):
        if L is None:
            continue
        # diagnostics on the central column only (columns are near-copies)
        mid = L[len(L) // 2]
        if method == "rays" and mid[0].size > 1:
            _warn_undersampled_shape(res, k, mid[0], mid[1], mid[2])
        # each sub-curve carries an equal share of the line's unit flux
        for cx, cy, w, clouds in L:
            # fine samples along the line (arc length, ~pitch/2 steps)
            if cx.size > 1:
                ds = np.hypot(np.diff(cx), np.diff(cy))
                t = np.concatenate(([0.0], np.cumsum(ds)))
                n_fine = max(cx.size,
                             int(np.ceil(t[-1] / (pitch_f / 2.0))) + 1)
                tf = np.linspace(0.0, t[-1], n_fine)
                cxf = np.interp(tf, t, cx)
                cyf = np.interp(tf, t, cy)
                wf = np.interp(tf, t, w)
                seg = np.clip(np.searchsorted(t, tf, side="right") - 1,
                              0, cx.size - 2)
                frac = np.where(np.diff(t)[seg] > 0,
                                (tf - t[seg]) / np.diff(t)[seg], 0.0)
            else:
                cxf, cyf, wf = cx, cy, w
                seg = np.zeros(1, dtype=int)
                frac = np.zeros(1)
            flux = 1.0 / (cxf.size * len(L))

            if method == "gaussian":
                from scipy.special import erf

                for j in range(cxf.size):
                    sigma = max(wf[j] / np.sqrt(2.0), 1e-9)
                    half = 4.0 * sigma
                    i0 = max(0, np.searchsorted(x_edges, cxf[j] - half) - 1)
                    i1 = min(n_px, np.searchsorted(x_edges, cxf[j] + half) + 1)
                    j0 = max(0, np.searchsorted(y_edges, cyf[j] - half) - 1)
                    j1 = min(n_py, np.searchsorted(y_edges, cyf[j] + half) + 1)
                    if i0 >= i1 or j0 >= j1:
                        continue
                    sq = sigma * np.sqrt(2.0)
                    fx = 0.5 * np.diff(erf((x_edges[i0:i1 + 1] - cxf[j]) / sq))
                    fy = 0.5 * np.diff(erf((y_edges[j0:j1 + 1] - cyf[j]) / sq))
                    image[j0:j1, i0:i1] += flux * np.outer(fy, fx)
            else:
                def deposit(pts_x, pts_y, weight):
                    ip = np.floor((pts_x - x_min) / pitch_f).astype(int)
                    jp = np.floor((pts_y - y_min) / pitch_f).astype(int)
                    ok = (ip >= 0) & (ip < n_pxf) & (jp >= 0) & (jp < n_pyf)
                    np.add.at(image, (jp[ok], ip[ok]), weight)

                for j in range(cxf.size):
                    a, f = int(seg[j]), float(frac[j])
                    for cloud, wgt in ((clouds[a], (1.0 - f)),
                                       (clouds[min(a + 1, len(clouds) - 1)],
                                        f)):
                        if wgt <= 0.0 or cloud.shape[0] == 0:
                            continue
                        deposit(cloud[:, 0] + cxf[j], cloud[:, 1] + cyf[j],
                                flux * wgt / cloud.shape[0])

    if ov > 1:
        image = image.reshape(n_py, ov, n_px, ov).sum(axis=(1, 3))

    x_centers = x_min + (np.arange(n_px) + 0.5) * pitch_mm
    y_centers = y_min + (np.arange(n_py) + 0.5) * pitch_mm
    attrs = {
        "units": "fraction of line flux per pixel",
        "pixel_pitch_um": float(pixel_pitch.to(u.um).value),
        "method": method,
        "wavelengths_nm": [lam * 1000.0 for lam in res.wavelengths_um],
        "line_x_mm": [float(np.mean(np.concatenate([C[0] for C in L])))
                      if L is not None else np.nan for L in lines],
        "line_y_mm": [float(np.mean(np.concatenate([C[1] for C in L])))
                      if L is not None else np.nan for L in lines],
        "detector_center_mm": list(detector_center_mm),
        "origin": "Zemax image-surface vertex (local frame)",
    }
    return image, x_centers, y_centers, attrs


def _warn_undersampled_shape(res: SpotScanResult, k: int,
                             cx: np.ndarray, cy: np.ndarray,
                             w: np.ndarray,
                             rms_jump_tol: float = 0.3,
                             bend_tol: float = 0.5) -> None:
    """Warn where the slit sampling is too coarse for shape morphing.

    Cross-fading neighboring ray clouds assumes the spot evolves roughly
    linearly between slit samples.  Two symptoms of violation:
      * fractional RMS jump between adjacent samples exceeding
        `rms_jump_tol` (aberration content changing fast);
      * centroid bending: second difference exceeding `bend_tol` x the
        local RMS (the line curves significantly between samples).
    Both produce ghost/doubling artifacts in the render; the fix is more
    real ray data there — pass denser `slit_points` around the reported
    field positions.
    """
    slit = res.y_field if res.y_field.size > 1 else res.x_field
    flags = np.zeros(cx.size, dtype=bool)
    jump = np.abs(np.diff(w)) / np.maximum(w[:-1], 1e-12)
    flags[:-1] |= jump > rms_jump_tol
    if cx.size > 2:
        bend = np.hypot(np.diff(cx, 2), np.diff(cy, 2))
        flags[1:-1] |= bend > bend_tol * np.maximum(w[1:-1], 1e-12)
    if flags.any() and slit.size == cx.size:
        logger.warning(
            "render: line %d (%.4g um): spot shape/position changes "
            "faster than the slit sampling near field = %s %s; expect "
            "morphing artifacts there — add denser slit_points in "
            "those regions",
            k + 1, res.wavelengths_um[k],
            np.array2string(slit[flags], precision=3, threshold=8),
            res.unit_name,
        )


def _image_secondary_coords(res: SpotScanResult, x_c: np.ndarray,
                            y_c: np.ndarray, attrs: dict) -> dict:
    """Non-dimension coordinates for the detector-image DataArray.

    * field coordinate: the slit field angle/height mapped onto the image
      dimension the slit runs along, via inverse interpolation of the
      MEAN centroid relation (averaged over lines and slit-width
      columns).  Per-line mappings differ by smile/keystone; the maximum
      line-to-line spread is reported in attrs["field_coord_spread_mm"]
      so the approximation is quantified, not hidden.  NaN outside the
      mapped range.
    * wavelength_nm: the dispersion solution along the dispersion image
      dimension, interpolated between line centers (>= 2 lines needed);
      NaN outside the outermost lines.
    """
    extra: dict = {}
    # ---- slit field coordinate -------------------------------------------
    slit_y = res.y_field.size >= res.x_field.size
    sf = res.y_field if slit_y else res.x_field
    if sf.size >= 2:
        avg_axes = (0, 2) if slit_y else (0, 1)
        px = np.nanmean(res.centroid_x_um, axis=avg_axes) / 1000.0
        py = np.nanmean(res.centroid_y_um, axis=avg_axes) / 1000.0
        pos, pix, dim = ((py, y_c, "y") if np.ptp(py) >= np.ptp(px)
                         else (px, x_c, "x"))
        d = np.diff(pos)
        if np.all(d > 0) or np.all(d < 0):
            p_srt, f_srt = ((pos, sf) if d[0] > 0
                            else (pos[::-1], sf[::-1]))
            vals = np.interp(pix, p_srt, f_srt)
            vals[(pix < p_srt[0]) | (pix > p_srt[-1])] = np.nan
            name = "field_y" if slit_y else "field_x"
            extra[name] = (dim, vals)
            # line-to-line spread of the mapping (smile/keystone)
            per_line = (res.centroid_y_um if dim == "y"
                        else res.centroid_x_um) / 1000.0
            per_line = (per_line if slit_y
                        else np.swapaxes(per_line, 1, 2))
            spread = np.nanmax(np.nanmax(per_line, axis=(0, 2))
                               - np.nanmin(per_line, axis=(0, 2)))
            attrs["field_coord_units"] = res.unit_name
            attrs["field_coord_dim"] = dim
            attrs["field_coord_spread_mm"] = float(spread)
        else:
            logger.info("secondary coords: mean centroid not monotonic "
                        "along the slit; field coordinate omitted")
    # ---- wavelength coordinate -------------------------------------------
    lx = np.asarray(attrs.get("line_x_mm", []), dtype=float)
    ly = np.asarray(attrs.get("line_y_mm", []), dtype=float)
    lam = np.asarray(attrs.get("wavelengths_nm", []), dtype=float)
    good = np.isfinite(lx) & np.isfinite(ly)
    if good.sum() >= 2:
        lx, ly, lam = lx[good], ly[good], lam[good]
        pos, pix, dim = ((lx, x_c, "x") if np.ptp(lx) >= np.ptp(ly)
                         else (ly, y_c, "y"))
        if np.unique(pos).size == pos.size:
            order = np.argsort(pos)
            vals = np.interp(pix, pos[order], lam[order])
            vals[(pix < pos[order][0]) | (pix > pos[order][-1])] = np.nan
            extra["wavelength_nm"] = (dim, vals)
            attrs["wavelength_coord_dim"] = dim
            attrs["wavelength_coord_note"] = (
                "piecewise-linear between line centers; NaN outside")
    return extra


def plot_detector_image(da, annotate: bool = True,
                        savepath: Optional[str] = None, ax=None,
                        cmap: str = "inferno"):
    """Plot a detector image produced by SpotScanResult.to_image().

    Pure presentation: all data and metadata come from the DataArray
    (values, x/y pixel-center coordinates in mm, and .attrs).  For quick
    looks, xarray's own da.plot() also works; this adds equal aspect,
    correct pixel extent, and wavelength annotations at line centers.
    """
    import matplotlib.pyplot as plt

    x_c = np.asarray(da["x"].values, dtype=float)
    y_c = np.asarray(da["y"].values, dtype=float)
    pitch = float(da.attrs.get("pixel_pitch_um", 0.0)) / 1000.0
    half = pitch / 2.0 if pitch else 0.0
    extent = (x_c[0] - half, x_c[-1] + half, y_c[0] - half, y_c[-1] + half)

    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 6.5))
    else:
        fig = ax.figure
    im = ax.imshow(np.asarray(da.values), origin="lower", cmap=cmap,
                   extent=extent, aspect="equal", interpolation="nearest")
    fig.colorbar(im, ax=ax,
                 label=da.attrs.get("units", "flux per pixel"))
    if annotate:
        lams = da.attrs.get("wavelengths_nm", [])
        lxs = da.attrs.get("line_x_mm", [])
        lys = da.attrs.get("line_y_mm", [])
        for lam, lx, ly in zip(lams, lxs, lys):
            if np.isfinite(lx) and np.isfinite(ly):
                ax.annotate(f"{lam:.1f} nm", xy=(lx, ly),
                            xytext=(6, 0), textcoords="offset points",
                            color="w", fontsize=8, ha="left",
                            va="center", rotation=90)
    ax.set_xlabel("image plane X [mm]")
    ax.set_ylabel("image plane Y [mm]")
    ax.set_title(f"Simulated detector image "
                 f"({da.attrs.get('pixel_pitch_um', '?')} um pixels, "
                 f"{da.attrs.get('method', '?')})")
    fig.tight_layout()
    if savepath:
        fig.savefig(savepath, dpi=150)
        print(f"detector image saved to {savepath}")
    return fig, ax
