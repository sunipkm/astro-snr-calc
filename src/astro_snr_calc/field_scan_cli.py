from __future__ import annotations
import logging
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Optional

import sys
if sys.platform != 'win32':
    raise RuntimeError(
        "This script requires Windows platform with ZOS-API available.")

import astropy.units as u
import numpy as np

from serde_dataclass import TomlDataclass

from .zemax_field_scan import (
    FieldScanConfig,
    SpotScanResult,
    plot_detector_image,
    plot_spectral_lines,
)


# ======================================================================
# Configuration dataclass (TOML-serializable)
# ======================================================================
@dataclass
class ScanConfig(TomlDataclass):
    """Image-plane spot scan configuration."""
    x: list = dc_field(
        default_factory=lambda: [1.0],
        metadata={"description": "X field range: one value V -> [-V,+V], "
                                 "or two values [LO, HI] in degrees"})
    y: list = dc_field(
        default_factory=lambda: [0.5],
        metadata={"description": "Y field range: one value V -> [-V,+V], "
                                 "or two values [LO, HI] in degrees"})
    nx: int = dc_field(default=20,
                       metadata={"description": "Number of X grid points"})
    ny: int = dc_field(default=30,
                       metadata={"description": "Number of Y grid points"})
    rings: int = dc_field(
        default=6,
        metadata={"description": "Hexapolar pupil rings (6 -> 127 rays)"})
    wavelengths: list = dc_field(
        default_factory=list,
        metadata={"description": "Wavelengths in nm; empty list = all defined in the file"})
    normalization: Optional[str] = dc_field(
        default=None,
        metadata={"description": "Field normalization: radial or rectangular"})
    pitch: Optional[float] = dc_field(
        default=None,
        metadata={"description": "Detector pixel pitch in um; required with --image"})
    detector: Optional[list] = dc_field(
        default=None,
        metadata={"description": "Detector size [W_MM, H_MM]"})
    oversample: int = dc_field(
        default=4,
        metadata={"description": "Sub-pixel deposition factor for --image"})
    n_workers: int = dc_field(
        default=1,
        metadata={"description": "Number of parallel workers for rendering spectral lines (default 1)"})
    save_rays: bool = dc_field(
        default=False,
        metadata={"description": "Save individual ray data (.rays) after scanning"})
    save_field: bool = dc_field(
        default=False,
        metadata={"description": "Save the rendered detector image (.field.nc) after scanning"})


# ======================================================================
# Command-line entry point
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
                    help=".zmx/.zos design file")
    ap.add_argument("--config", metavar="FILE",
                    help="TOML configuration file; CLI args override "
                    "individual fields (use --write-config to generate "
                    "a template)")
    ap.add_argument("--write-config", metavar="FILE",
                    help="write a template TOML configuration to FILE and exit")
    ap.add_argument("--save-rays", action="store_true",
                    help="save individual ray data for later inspection")
    ap.add_argument("--save-field", action="store_true",
                    help="save the field image data for later inspection")
    ap.add_argument("-x", type=float, nargs="+", default=None,
                    metavar="V",
                    help="X field range: one value V -> [-V, +V], or two "
                    "values LO HI (default: -1 1)")
    ap.add_argument("-y", type=float, nargs="+", default=None,
                    metavar="V",
                    help="Y field range: one value V -> [-V, +V], or two "
                    "values LO HI (default: -0.5 0.5)")
    ap.add_argument("-nx", type=int, default=None)
    ap.add_argument("-ny", type=int, default=None)
    ap.add_argument("--rings", type=int, default=None,
                    help="hexapolar pupil rings (default 6 -> 127 rays)")
    ap.add_argument("--waves", type=float, nargs="*", default=None,
                    metavar="NM",
                    help="wavelengths in nm, e.g. --waves 486.1 550 656.3 "
                    "(default: all wavelengths defined in the file)")
    ap.add_argument("--normalization", choices=["radial", "rectangular"],
                    default=None, help="set field normalization first")
    ap.add_argument("--plot", action="store_true",
                    help="save a polychromatic RMS map to this file")
    ap.add_argument("--lines", action="store_true",
                    help="spectrograph mode: save the spectral lines as "
                    "they land on the image plane (use with --waves and "
                    "a slit-like grid, e.g. --nx 1)")
    ap.add_argument("--detector", type=float, nargs=2, default=None,
                    metavar=("W_MM", "H_MM"),
                    help="detector outline for --lines / extent for "
                    "--image (IMX267: 14.19 10.38)")
    ap.add_argument("--oversample", type=int, default=None, metavar="N",
                    help="sub-pixel deposition factor for --image "
                    "(anti-aliasing of sharp lines; default 4)")
    ap.add_argument("--pitch", type=float, default=None, metavar="UM",
                    help="detector pixel pitch in um (IMX267: 3.45); "
                    "required with --image")
    ap.add_argument("--image", action="store_true",
                    help="spectrograph mode: render the lines as a "
                    "pixel-sampled detector intensity image, spot flux "
                    "distributed over pixels")
    ap.add_argument("--n-workers", type=int, default=None, metavar="N",
                    help="number of parallel workers for rendering spectral lines "
                    "(default 1)")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="[%(asctime)s] %(name)s: %(message)s", datefmt="%H:%M:%S",
    )

    # Load or create base configuration, then apply CLI overrides
    if args.config:
        scan_cfg = ScanConfig.from_toml(
            Path(args.config).read_text(encoding="utf-8"))
    else:
        scan_cfg = ScanConfig()

    if args.write_config:
        Path(args.write_config).write_text(
            scan_cfg.to_toml(), encoding="utf-8")
        print(f"Configuration template written to {args.write_config}")
        return

    if args.zemax_file is None:
        ap.error("zemax_file is required unless --write-config is used")

    x_vals = args.x if args.x is not None else scan_cfg.x
    y_vals = args.y if args.y is not None else scan_cfg.y
    nx = args.nx if args.nx is not None else scan_cfg.nx
    ny = args.ny if args.ny is not None else scan_cfg.ny
    rings = args.rings if args.rings is not None else scan_cfg.rings
    waves = args.waves if args.waves is not None else scan_cfg.wavelengths
    norm = args.normalization if args.normalization is not None else scan_cfg.normalization
    pitch = args.pitch if args.pitch is not None else scan_cfg.pitch
    detector = args.detector if args.detector is not None else scan_cfg.detector
    oversample = args.oversample if args.oversample is not None else scan_cfg.oversample
    n_workers = args.n_workers if args.n_workers is not None else scan_cfg.n_workers
    save_rays  = args.save_rays  or scan_cfg.save_rays
    save_field = args.save_field or scan_cfg.save_field
    render_image = bool(args.image)

    zemax_file = Path(args.zemax_file).absolute()

    def to_range(vals, name):
        if len(vals) == 1:
            return (-vals[0] * u.deg, vals[0] * u.deg)
        if len(vals) == 2:
            return (vals[0] * u.deg, vals[1] * u.deg)
        ap.error(f"--{name} takes one value (+/-V) or two (LO HI)")

    cfg = FieldScanConfig(
        x_range=to_range(x_vals, "x"), n_x=nx,
        y_range=to_range(y_vals, "y"), n_y=ny,
        wavelengths=(None if not waves
                     else tuple(w * u.nm for w in waves)),
        pupil_rings=rings,
        normalization=norm,
        keep_rays=render_image or save_field,   # true spot shapes for --image / --save-field
    )
    from .zemax_iface import ZOSConnection

    zos = ZOSConnection()
    try:
        zos.load(zemax_file)
        res = cfg.scan_image_plane(zos)
    finally:
        zos.close()

    if save_rays:
        outf = zemax_file.with_suffix(".rays")
        res.save(outf)
        print(f"Rays saved to {outf}")

    _print_report(res)
    if args.plot:
        outf = zemax_file.with_suffix(".plot.png")
        _plot_report(res, str(outf))
    if args.lines:
        outf = zemax_file.with_suffix(".lines.png")
        plot_spectral_lines(
            res, savepath=str(outf),
            detector_mm=(
                tuple(detector)
                if detector else None
            ),
        )
        print(f"Spectral lines saved to {outf}")
    if render_image or save_field:
        if pitch is None:
            ap.error("--image requires --pitch (um)")
        da = res.to_image(
            pixel_pitch=pitch * u.um,
            detector_mm=(tuple(detector) if detector else None),
            oversample=oversample,
            n_workers=n_workers,
        )
        if render_image:
            outf = zemax_file.with_suffix(".image.png")
            plot_detector_image(da, savepath=str(outf))
            print(f"Image saved to {outf}")
        if save_field:
            outf = zemax_file.with_suffix(".field.nc")
            da.to_netcdf(outf)
            print(f"Field saved to {outf}")


if __name__ == "__main__":
    main()
