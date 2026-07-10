# ======================================================================
# Self-test: mock ZOS-API implementing a paraxial lens + known blur
# ======================================================================
from __future__ import annotations
import logging
from typing import Optional

import astropy.units as u
import numpy as np

from .zemax_field_scan import (
    FieldScanConfig, 
    SpotScanResult,
    hexapolar_pupil, 
    plot_detector_image, 
    plot_spectral_lines,
    scan_spectral_lines, 
    set_field_normalization, 
    logger
)


_MOCK_F_MM = 119.687     # focal length of the mock lens
_MOCK_RMAX_DEG = 2.0     # maximum defined (radial) field of the mock
_MOCK_S0_MM = 0.010      # blur slope: dx = s * Px, s = S0 * (1 + 2 r_n^2)
_MOCK_DISP_MM = 0.5      # mock dispersion: x shift per wavelength number


def _mock_blur_slope(hx: float, hy: float) -> float:
    return _MOCK_S0_MM * (1.0 + 2.0 * (hx ** 2 + hy ** 2))


def _build_mock_zos():
    """A fake ZOSConnection wrapping an ideal thin lens with an injected,
    field-dependent geometric blur — so scan results have closed-form
    expected values."""

    class Obj:
        def __init__(self, **kw):
            self.__dict__.update(kw)

    class NormUnpol:
        def __init__(self):
            self.rays, self.i = [], 0

        def AddRay(self, wave, hx, hy, px, py, opd):
            self.rays.append((wave, hx, hy, px, py))

        def StartReadingResults(self):
            self.i = 0

        def ReadNextResult(self):
            wave, hx, hy, px, py = self.rays[self.i]
            self.i += 1
            s = _mock_blur_slope(hx, hy)
            x = (_MOCK_F_MM * np.tan(np.deg2rad(hx * _MOCK_RMAX_DEG))
                 + s * px + _MOCK_DISP_MM * (wave - 1))   # "grating"
            y = _MOCK_F_MM * np.tan(np.deg2rad(hy * _MOCK_RMAX_DEG)) + s * py
            return (True, self.i, 0, 0, x, y, 0, 0, 0, 1, 0, 0, 1, 0, 1)

    class Tool:
        def CreateNormUnpol(self, n, rays_type, surf):
            return NormUnpol()

        def RunAndWaitForCompletion(self):
            pass

        def Close(self):
            pass

    class Fields:
        NumberOfFields = 2
        Normalization = "Radial"

        @staticmethod
        def GetFieldType():
            return "Angle"

        @staticmethod
        def GetField(i):
            return (Obj(X=0.0, Y=0.0) if i == 1
                    else Obj(X=0.0, Y=_MOCK_RMAX_DEG))

    class Waves:
        def __init__(self):
            self._w = [0.55]

        @property
        def NumberOfWavelengths(self):
            return len(self._w)

        def GetWavelength(self, i):
            return Obj(Wavelength=self._w[i - 1])

        def AddWavelength(self, um, weight):
            self._w.append(float(um))
            return Obj(Wavelength=float(um))

        def RemoveWavelength(self, i):
            del self._w[i - 1]
            return True

    class ZOSAPI:
        class SystemData:
            FieldNormalizationType = Obj(Radial="Radial",
                                         Rectangular="Rectangular")

        class Tools:
            class RayTrace:
                RaysType = Obj(Real=0)
                OPDMode = type("OPDMode", (), {"None": 0})

    class MockZOS:
        pass

    mz = MockZOS()
    mz.ZOSAPI = ZOSAPI
    mz.system = Obj(
        SystemData=Obj(Fields=Fields, Wavelengths=Waves()),
        LDE=Obj(NumberOfSurfaces=7),
        Tools=Obj(OpenBatchRayTrace=Tool),
    )
    return mz


def self_test() -> None:
    """Validate the scan pipeline against the mock (no OpticStudio).

    Checks: centroid = f*tan(theta) to <1e-6 um, RMS matches the injected
    blur analytically, chunked bookkeeping, interpolation, worst-field
    location, normalization setter round-trip.
    """
    cfg = FieldScanConfig(
        x_range=(-1 * u.deg, 1 * u.deg), n_x=5,
        y_range=(-0.5 * u.deg, 0.5 * u.deg), n_y=7,
        # 550 nm reuses the mock's 0.55 um table entry; 486 nm must be
        # temporarily added and removed again
        wavelengths=(550 * u.nm, 486 * u.nm),
        pupil_rings=4,
        batch_chunk=400,            # small -> forces multiple chunks
        normalization="radial",     # exercises the setter path too
    )
    mock = _build_mock_zos()
    res = cfg.scan_image_plane(mock)
    assert res.wave_indices == [1, 2] and len(res.wavelengths_um) == 2
    assert abs(res.wavelengths_um[1] - 0.486) < 1e-12
    assert mock.system.SystemData.Wavelengths.NumberOfWavelengths == 1, \
        "temporary wavelength was not removed"

    px, py = hexapolar_pupil(4)
    r2_mean = float(np.mean(px ** 2 + py ** 2))
    ix, iy = 3, 5
    exp_cx = _MOCK_F_MM * np.tan(np.deg2rad(res.x_field[ix])) * 1000.0
    assert abs(res.centroid_x_um[0, iy, ix] - exp_cx) < 1e-6, "centroid"
    hx = res.x_field[ix] / _MOCK_RMAX_DEG
    hy = res.y_field[iy] / _MOCK_RMAX_DEG
    exp_rms = _mock_blur_slope(hx, hy) * np.sqrt(r2_mean) * 1000.0
    assert abs(res.rms_um[0, iy, ix] - exp_rms) / exp_rms < 1e-9, "RMS"
    # dispersion: line centroids separated by _MOCK_DISP_MM per wave step
    sep = res.centroid_x_um[1, iy, ix] - res.centroid_x_um[0, iy, ix]
    assert abs(sep - _MOCK_DISP_MM * 1000.0) < 1e-6, "dispersion"
    # pooled poly RMS = quadrature of per-line RMS and half-separation
    exp_poly = np.sqrt(exp_rms ** 2 + (sep / 2.0) ** 2)
    assert abs(res.rms_poly_um[iy, ix] - exp_poly) / exp_poly < 1e-9, \
        "poly RMS with dispersion"
    assert res.n_valid.min() == px.size == res.n_valid.max(), "bookkeeping"
    assert abs(res.rms_at(res.x_field[ix] * u.deg, res.y_field[iy] * u.deg,
                          wavelength=550 * u.nm)
               - exp_rms) < 1e-9, "interpolation (single line)"
    assert abs(res.rms_at(res.x_field[ix] * u.deg, res.y_field[iy] * u.deg)
               - exp_poly) < 1e-9, "interpolation (polychromatic)"
    assert abs(res.rms_at(res.x_field[ix] * u.deg, res.y_field[iy] * u.deg,
                          wavelength=486 * u.nm)
               - exp_rms) < 1e-9, "wavelength selection"
    try:
        res.rms_at(0 * u.deg, 0 * u.deg, wavelength=700 * u.nm)
        raise AssertionError("unscanned wavelength not rejected")
    except ValueError:
        pass
    wx, wy = res.worst_field()
    assert (abs(abs(wx.value) - 1.0) < 1e-12
            and abs(abs(wy.value) - 0.5) < 1e-12), "worst field"
    mz = _build_mock_zos()
    assert set_field_normalization(mz, "rectangular") == "Rectangular"
    assert set_field_normalization(mz, "RADIAL") == "Radial"

    # spectral-line scan: 1-D slit sweep, three "lines" on the detector
    lines = scan_spectral_lines(
        _build_mock_zos(),
        wavelengths=(486 * u.nm, 550 * u.nm, 656 * u.nm),
        slit_range=(-0.5 * u.deg, 0.5 * u.deg), n_slit=9,
        slit_axis="y", pupil_rings=3,
    )
    assert lines.rms_um.shape == (3, 9, 1)
    lx = np.nanmean(lines.centroid_x_um, axis=(1, 2))
    exp_lx = _MOCK_DISP_MM * 1000.0 * (np.asarray(lines.wave_indices) - 1)
    assert np.allclose(lx, exp_lx), "line positions on the detector"
    import matplotlib
    matplotlib.use("Agg", force=True)
    fig, _ = plot_spectral_lines(lines, detector_mm=(14.19, 10.38))
    assert len(fig.axes[0].lines) >= 3, "one curve per spectral line"

    # true-spot-shape rendering: the mock spot is a uniformly scaled
    # hexapolar disk (NOT a Gaussian); a single-point "line" rendered at
    # fine pitch must reproduce the ray cloud's second moment, and the
    # rays must be recoverable via rays_at.
    one = scan_spectral_lines(
        _build_mock_zos(), wavelengths=(550 * u.nm,),
        slit_range=(0.2 * u.deg, 0.2 * u.deg), n_slit=1, pupil_rings=6,
    )
    cloud = one.rays_at(0, 0, 0)
    assert cloud.shape == (127, 2), "rays_at"
    da1 = one.to_image(pixel_pitch=2 * u.um)
    img1 = np.asarray(da1.values)
    assert da1.attrs["method"] == "rays" and da1.dims == ("y", "x")
    assert abs(img1.sum() - 1.0) < 1e-12, "ray flux conservation"
    pitch_mm = da1.attrs["pixel_pitch_um"] / 1000.0
    xc = np.asarray(da1["x"].values, dtype=float)
    col = img1.sum(axis=0)
    mu = np.sum(xc * col) / col.sum()
    var_img = np.sum((xc - mu) ** 2 * col) / col.sum()
    var_rays = np.var(cloud[:, 0] / 1000.0)
    # image variance = ray variance + pixel binning variance (pitch^2/12)
    assert abs(var_img - (var_rays + pitch_mm ** 2 / 12.0)) \
        < 0.02 * var_rays, "spot shape second moment"

    # oversampled deposition: identical flux, same second moment
    da_ov = one.to_image(pixel_pitch=2 * u.um, oversample=8)
    img_ov = np.asarray(da_ov.values)
    assert abs(img_ov.sum() - 1.0) < 1e-12, "oversampled flux"
    assert img_ov.shape == img1.shape, "oversampling keeps native grid"

    # non-uniform slit sampling (densified center)
    pts = tuple(v * u.deg for v in
                (-0.5, -0.3, -0.1, -0.05, 0.0, 0.05, 0.1, 0.3, 0.5))
    nonuni = scan_spectral_lines(
        _build_mock_zos(), wavelengths=(550 * u.nm,),
        slit_points=pts, pupil_rings=3)
    assert nonuni.y_field.size == 9
    assert not np.allclose(np.diff(nonuni.y_field),
                           np.diff(nonuni.y_field)[0]), "non-uniform axis"
    assert np.all(np.isfinite(nonuni.rms_um)), "non-uniform scan traces"

    # undersampling diagnostic: coarse slit over strong blur gradient
    records = []
    h = logging.Handler()
    h.emit = lambda rec: records.append(rec.getMessage())
    logger.addHandler(h)
    try:
        coarse = scan_spectral_lines(
            _build_mock_zos(), wavelengths=(550 * u.nm,),
            slit_range=(-1.9 * u.deg, 1.9 * u.deg), n_slit=3,
            pupil_rings=3)
        coarse.to_image(pixel_pitch=10 * u.um)
    finally:
        logger.removeHandler(h)
    assert any("faster than the slit sampling" in m for m in records), \
        "undersampling warning"

    # finite slit width: a 2-D scan renders as parallel sub-curves with
    # shared flux, not a raveled zigzag
    wide = FieldScanConfig(
        x_range=(-0.02 * u.deg, 0.02 * u.deg), n_x=3,
        y_range=(-0.4 * u.deg, 0.4 * u.deg), n_y=9,
        wavelengths=(550 * u.nm,), pupil_rings=3, keep_rays=True,
        normalization="radial")
    wres = wide.scan_image_plane(_build_mock_zos())
    da_w = wres.to_image(pixel_pitch=10 * u.um)
    img_w = np.asarray(da_w.values)
    assert abs(img_w.sum() - 1.0) < 1e-12, "2-D scan flux conservation"
    # x footprint must span the three columns (~2*f*tan(0.02deg) ~ 84 um),
    # far wider than a single column's spot alone
    xs = np.asarray(da_w["x"].values, dtype=float)
    prof = img_w.sum(axis=0)
    lit = xs[prof > prof.max() * 1e-3]
    assert (lit.max() - lit.min()) > 0.06, "slit width rendered"

    # rays ndarray: field coords + landing positions per ray
    r = nonuni.rays
    assert isinstance(r, np.ndarray) and r.shape[1] == 5
    assert set(np.round(np.unique(r[:, 2]), 6)) <= \
        set(np.round(nonuni.y_field, 6)), "casting fields stored"
    assert np.array_equal(nonuni.rays_at(0, 4, 0),
                          r[(r[:, 0] == 0)
                            & np.isclose(r[:, 2],
                                         nonuni.y_field[4])][:, 3:5])

    # concat: two half-slit scans == one full scan (field merge)
    full_pts = tuple(v * u.deg for v in np.linspace(-0.5, 0.5, 9))
    full = scan_spectral_lines(_build_mock_zos(),
                               wavelengths=(550 * u.nm,),
                               slit_points=full_pts, pupil_rings=3)
    ha = scan_spectral_lines(_build_mock_zos(), wavelengths=(550 * u.nm,),
                             slit_points=full_pts[:5], pupil_rings=3)
    hb = scan_spectral_lines(_build_mock_zos(), wavelengths=(550 * u.nm,),
                             slit_points=full_pts[5:], pupil_rings=3)
    merged = SpotScanResult.concat_scans([hb, ha])         # out of order on purpose
    assert np.allclose(merged.y_field, full.y_field)
    assert np.allclose(merged.rms_um, full.rms_um)
    assert np.allclose(merged.centroid_x_um, full.centroid_x_um)
    assert merged.rays.shape[0] == full.rays.shape[0]
    m_img = np.asarray(merged.to_image(pixel_pitch=25 * u.um).values)
    f_img = np.asarray(full.to_image(pixel_pitch=25 * u.um).values)
    assert np.allclose(m_img, f_img), "merged scan renders identically"

    # concat: wavelength merge on the same grid
    wa = scan_spectral_lines(_build_mock_zos(), wavelengths=(550 * u.nm,),
                             slit_points=full_pts, pupil_rings=3)
    wb = scan_spectral_lines(_build_mock_zos(), wavelengths=(486 * u.nm,),
                             slit_points=full_pts, pupil_rings=3)
    wl = SpotScanResult.concat_scans([wa, wb])
    assert wl.rms_um.shape[0] == 2 and len(wl.wavelengths_um) == 2
    sep = (np.nanmean(wl.centroid_x_um[1]) - np.nanmean(
        wl.centroid_x_um[0]))
    assert abs(sep - _MOCK_DISP_MM * 1000.0) < 1e-6, \
        "wavelength-merged dispersion"
    assert np.all(np.isfinite(wl.rms_poly_um)), "poly RMS from rays"

    # secondary coordinates on the detector image
    da3 = lines.to_image(pixel_pitch=25 * u.um)
    fy = np.asarray(da3["field_y"].values, dtype=float)
    yc3 = np.asarray(da3["y"].values, dtype=float)
    tgt = _MOCK_F_MM * np.tan(np.deg2rad(0.3)) / 1000.0 * 1000.0  # mm
    j = int(np.argmin(np.abs(yc3 - tgt)))
    assert abs(fy[j] - 0.3) < 5e-3, "field_y secondary coordinate"
    assert np.isnan(fy[0]) and np.isnan(fy[-1]), "field_y NaN outside"
    assert da3.attrs["field_coord_units"] == "deg"
    wl = np.asarray(da3["wavelength_nm"].values, dtype=float)
    xc3 = np.asarray(da3["x"].values, dtype=float)
    # mock line centers: 550 nm at 0 mm, 486 nm at 0.5 mm, 656 at 1.0 mm
    assert abs(wl[np.argmin(np.abs(xc3 - 0.5))] - 486.0) < 5.0
    assert abs(wl[np.argmin(np.abs(xc3 - 0.25))] - 518.0) < 5.0, \
        "wavelength coordinate interpolates between line centers"

    # persistence + in-memory composition: save both halves, load them,
    # and concatenate with `+`; must equal the direct merge exactly
    import tempfile, os
    with tempfile.TemporaryDirectory() as td:
        pa, pb = os.path.join(td, "a.npz"), os.path.join(td, "b.npz")
        ha.save(pa)
        hb.save(pb)
        la, lb = SpotScanResult.load(pa), SpotScanResult.load(pb)
        assert np.allclose(la.rms_um, ha.rms_um)
        assert np.array_equal(la.rays, ha.rays)
        assert la.unit_name == ha.unit_name
        summed = lb + la                       # operator form
        assert np.allclose(summed.rms_um, merged.rms_um)
        assert np.allclose(summed.centroid_x_um, merged.centroid_x_um)
        assert np.array_equal(np.sort(summed.rays, axis=0),
                              np.sort(merged.rays, axis=0))
        assert np.allclose(sum([la, lb]).rms_um, merged.rms_um)  # sum()

    # detector offset: grid recenters, deposited flux pattern unchanged
    # (mock spot sits at y ~= f*tan(0.2 deg) ~= 0.418 mm off the vertex)
    da_off = one.to_image(pixel_pitch=2 * u.um, detector_mm=(0.5, 0.5),
                          detector_center_mm=(0.05, 0.42))
    xo = np.asarray(da_off["x"].values, dtype=float)
    assert abs((xo[0] + xo[-1]) / 2.0 - 0.05) < 2e-3, "detector center"
    assert abs(np.asarray(da_off.values).sum() - 1.0) < 1e-12, \
        "flux on the offset detector"
    # a detector centered on the vertex misses this off-axis spot
    da_miss = one.to_image(pixel_pitch=2 * u.um, detector_mm=(0.2, 0.2))
    assert np.asarray(da_miss.values).sum() < 1e-12, \
        "spot correctly clipped off a vertex-centered detector"

    # detector-image rendering: flux conservation + line positions
    da = lines.to_image(pixel_pitch=25 * u.um)
    img = np.asarray(da.values)
    assert abs(img.sum() - 3.0) < 1e-3, "flux conservation (3 unit lines)"
    xc = np.asarray(da["x"].values, dtype=float)          # pixel centers
    col = img.sum(axis=0)
    fig_img, _ = plot_detector_image(da)                  # plot layer
    assert fig_img.axes, "plot_detector_image produced no axes"
    exp_lx = _MOCK_DISP_MM * (np.asarray(lines.wave_indices) - 1)   # mm
    for lx0 in exp_lx:
        sel = np.abs(xc - lx0) < 0.2
        cen = float(np.sum(xc[sel] * col[sel]) / np.sum(col[sel]))
        assert abs(cen - lx0) < 25e-3 / 2, "line centroid on pixel grid"

    print(f"self-test: centroid {res.centroid_x_um[0, iy, ix]:.3f} um, "
          f"RMS {res.rms_um[0, iy, ix]:.4f} um "
          f"(expected {exp_rms:.4f}), worst field "
          f"({wx.value:g}, {wy.value:g}) deg")
    print("self-test: ALL CHECKS PASSED")


# ======================================================================
# Command-line test program
# ======================================================================
def _print_report(res: SpotScanResult) -> None:
    print(f"grid: {res.x_field.size} x {res.y_field.size} over "
          f"x=[{res.x_field[0]:g}, {res.x_field[-1]:g}] deg, "
          f"y=[{res.y_field[0]:g}, {res.y_field[-1]:g}] deg")
    for k, w in enumerate(res.wave_indices):
        r = res.rms_um[k]
        print(f"wave {w} ({res.wavelengths_um[k]:.4f} um): RMS spot "
              f"min {np.nanmin(r):.3f} / median {np.nanmedian(r):.3f} / "
              f"max {np.nanmax(r):.3f} um")
    rp = res.rms_poly_um
    print(f"polychromatic: RMS spot min {np.nanmin(rp):.3f} / "
          f"median {np.nanmedian(rp):.3f} / max {np.nanmax(rp):.3f} um")
    wx, wy = res.worst_field()
    print(f"worst field: ({wx.value:g}, {wy.value:g}) deg -> "
          f"{np.nanmax(rp):.3f} um RMS")


def _plot_report(res: SpotScanResult, path: str) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5.5))
    pcm = ax.pcolormesh(res.x_field, res.y_field, res.rms_poly_um,
                        shading="auto", cmap="viridis")
    fig.colorbar(pcm, ax=ax, label="polychromatic RMS spot radius [um]")
    wx, wy = res.worst_field()
    ax.plot(wx.value, wy.value, "r+", ms=12, mew=2, label="worst field")
    ax.set_xlabel("field X [deg]")
    ax.set_ylabel("field Y [deg]")
    ax.set_title("Image-plane spot scan")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    print(f"plot saved to {path}")


def main(argv: Optional[list[str]] = None) -> None:
    import argparse

    ap = argparse.ArgumentParser(
        description="Image-plane spot scan via ZOS-API batch ray tracing."
    )
    ap.add_argument("zemax_file", nargs="?",
                    help=".zmx/.zos design (omit with --self-test)")
    ap.add_argument("--self-test", action="store_true",
                    help="validate against the built-in mock lens; "
                    "no OpticStudio needed")
    ap.add_argument("--x", type=float, nargs="+", default=[1.0],
                    metavar="V",
                    help="X field range: one value V -> [-V, +V], or two "
                    "values LO HI (default: -1 1)")
    ap.add_argument("--y", type=float, nargs="+", default=[0.5],
                    metavar="V",
                    help="Y field range: one value V -> [-V, +V], or two "
                    "values LO HI (default: -0.5 0.5)")
    ap.add_argument("--nx", type=int, default=20)
    ap.add_argument("--ny", type=int, default=30)
    ap.add_argument("--rings", type=int, default=6,
                    help="hexapolar pupil rings (default 6 -> 127 rays)")
    ap.add_argument("--waves", type=float, nargs="*", default=None,
                    metavar="NM",
                    help="wavelengths in nm, e.g. --waves 486.1 550 656.3 "
                    "(default: all wavelengths defined in the file)")
    ap.add_argument("--normalization", choices=["radial", "rectangular"],
                    default=None, help="set field normalization first")
    ap.add_argument("--plot", metavar="PNG",
                    help="save a polychromatic RMS map to this file")
    ap.add_argument("--lines", metavar="PNG",
                    help="spectrograph mode: save the spectral lines as "
                    "they land on the image plane (use with --waves and "
                    "a slit-like grid, e.g. --nx 1)")
    ap.add_argument("--detector", type=float, nargs=2, default=None,
                    metavar=("W_MM", "H_MM"),
                    help="detector outline for --lines / extent for "
                    "--image (IMX267: 14.19 10.38)")
    ap.add_argument("--oversample", type=int, default=4, metavar="N",
                    help="sub-pixel deposition factor for --image "
                    "(anti-aliasing of sharp lines; default 4)")
    ap.add_argument("--pitch", type=float, default=None, metavar="UM",
                    help="detector pixel pitch in um (IMX267: 3.45); "
                    "required with --image")
    ap.add_argument("--image", metavar="PNG",
                    help="spectrograph mode: render the lines as a "
                    "pixel-sampled detector intensity image, spot flux "
                    "distributed over pixels")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    )

    if args.self_test:
        self_test()
        return
    if not args.zemax_file:
        ap.error("provide a Zemax file, or use --self-test")

    def to_range(vals, name):
        if len(vals) == 1:
            return (-vals[0] * u.deg, vals[0] * u.deg)
        if len(vals) == 2:
            return (vals[0] * u.deg, vals[1] * u.deg)
        ap.error(f"--{name} takes one value (+/-V) or two (LO HI)")

    cfg = FieldScanConfig(
        x_range=to_range(args.x, "x"), n_x=args.nx,
        y_range=to_range(args.y, "y"), n_y=args.ny,
        wavelengths=(None if args.waves is None
                     else tuple(w * u.nm for w in args.waves)),
        pupil_rings=args.rings,
        normalization=args.normalization,
        keep_rays=bool(args.image),   # true spot shapes for --image
    )
    # try:
    #     from .zemax_iface import ZOSConnection   # inside the package
    # except ImportError:
    #     from zemax_iface import ZOSConnection    # run as a plain script
    from astro_snr_calc.zemax_iface import ZOSConnection

    zos = ZOSConnection()
    try:
        zos.load(str(args.zemax_file))
        res = cfg.scan_image_plane(zos)
    finally:
        zos.close()

    _print_report(res)
    if args.plot:
        _plot_report(res, args.plot)
    if args.lines:
        plot_spectral_lines(res, savepath=args.lines,
                            detector_mm=(tuple(args.detector)
                                         if args.detector else None))
    if args.image:
        if args.pitch is None:
            ap.error("--image requires --pitch (um)")
        da = res.to_image(
            pixel_pitch=args.pitch * u.um,
            detector_mm=(tuple(args.detector) if args.detector else None),
            oversample=args.oversample,
        )
        plot_detector_image(da, savepath=args.image)


if __name__ == "__main__":
    main()