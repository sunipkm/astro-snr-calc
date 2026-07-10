"""SNRCalc – command-line SNR calculator for astronomical imagers.

Usage
-----
    SNRCalc <telescope.toml> [detector.toml] [options]

On Windows, a Zemax .zmx file can be passed instead of a TOML file to
drive the PSF computation via the ZOS-API.
"""
import sys
import logging

from .snr_calculator import (
    ExposureGrid, PhotometricBand,
    SNRResult, Sensor,
    SkyBrightness, Telescope,
    SNRMapPlotter,
)
from pathlib import Path
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import astropy.units as u

if sys.platform == "win32":
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(message)s",
    )
    logging.getLogger("zemax_iface").setLevel(logging.DEBUG)


def main():
    IMX267_UNCOOLED = Sensor(
        name="Sony IMX267 (uncooled)",
        pixel_pitch=3.45 * u.um,
        qe=0.64,
        read_noise_e=2.4,
        dark_rate=3.0 / u.s,
        full_well_e=10700.0,
    )

    parser = ArgumentParser(
        description="Calculate SNR for a given telescope and detector model, "
                    "optionally using Zemax PSF data."
    )
    parser.add_argument(
        "telescope",
        help="Telescope model TOML file (or .zmx file on Windows)"
    )
    parser.add_argument(
        'detector', nargs='?',
        help='Detector model TOML file (optional, default: built-in IMX267 uncooled model)'
    )
    parser.add_argument(
        '-p', '--pupil-sampling', default='S_32x32',
        help="Zemax pupil sampling (default: %(default)s)"
    )
    parser.add_argument(
        '-i', '--image-sampling', default='S_32x32',
        help="Zemax image sampling (default: %(default)s)"
    )
    parser.add_argument(
        '-f', '--field', nargs='*', type=int, dest='field_index', default=None,
        help="Evaluate only the specified field indices (default: all fields)"
    )
    parser.add_argument(
        '--no-plots', action='store_true',
        help="Do not show the resulting SNR plots"
    )
    parser.add_argument(
        '--no-save', action='store_true',
        help="Do not save the resulting SNR plots to files"
    )
    if sys.platform == "win32":
        parser.add_argument(
            '--huygens', action='store_true',
            help="Use Huygens PSF computation (slow, but accurate)"
        )
        parser.add_argument(
            '-e', '--as-extension', action='store_true',
            help="Connect to Zemax as an extension (requires interactive OpticStudio session)"
        )
    args = parser.parse_args()

    if args.detector is None:
        detector = IMX267_UNCOOLED
    else:
        detector = Sensor.load_toml(args.detector)

    zemax_file = None
    telescope = None

    try:
        config_file = Path(args.telescope)
        if config_file.is_file():
            telescope = Telescope.from_toml(
                config_file.read_text(encoding="utf-8"))
        else:
            raise ValueError(f"Telescope config file not found: {config_file}")
    except UnicodeDecodeError:
        zemax_file = Path(args.telescope)
    except Exception as e:
        print(f"Error loading telescope config: {e}")
        sys.exit(1)

    if zemax_file is not None and sys.platform != "win32":
        print(
            "Warning: Zemax file specified but Zemax is only supported on Windows."
        )
        sys.exit(1)

    if sys.platform != "win32" and telescope is not None:
        stem = "CUVISM-StarCam_v2"
        model = telescope.build(
            detector,
            PhotometricBand.johnson_v(),
            SkyBrightness.dark_sky(),
        )
        models = [("Analytic (on-axis)", model)]
    elif sys.platform == "win32" and zemax_file is not None:
        from .zemax_iface import ZemaxConfig
        config = ZemaxConfig(
            zemax_file=zemax_file,
            pupil_sampling=args.pupil_sampling,
            image_sampling=args.image_sampling,
            connect_as_extension=args.as_extension,
            psf_method="huygens" if args.huygens else "fft",
        )
        stem = zemax_file.stem
        indices = [] if args.field_index is None else args.field_index
        models = config.build(
            sensor=detector,
            band=PhotometricBand.johnson_v(),
            sky=SkyBrightness.dark_sky(),
            field_indices=indices,
        )
    else:
        print(
            "No Zemax file specified or Zemax not supported on this platform. "
            "Falling back to analytic model."
        )
        sys.exit(1)

    if len(models) > 1:
        print("\nField comparison:")
        for label, model in models:
            n_pix, ee = model.aperture()
            print(
                f"  {label}: peak fraction "
                f"{model.psf.peak_fraction:.3f}, aperture {n_pix} px, "
                f"EE {ee:.3f}"
            )
        print()

    plotter = SNRMapPlotter()
    for k, (label, model) in enumerate(models, start=1):
        print(f"=== {label} ===")
        print(model.summary())
        result = SNRResult.from_model(model, *ExposureGrid.default().arrays())
        suffix = f"_field{k}" if len(models) > 1 else ""
        plotter.plot(
            result,
            savepath=f"{stem}{suffix}.png" if not args.no_save else None,
        )

    if not args.no_plots:
        plt.show()
    else:
        print(
            "Plots were generated but not shown due to --no-plots flag. "
            "Saved plot files: "
            + ", ".join(f"{stem}_field{k}.png" for k in range(1,
                        len(models) + 1))
        )


if __name__ == "__main__":
    main()
