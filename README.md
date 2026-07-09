# astro-snr-calc

`astro-snr-calc` is a small Python library and command-line tool for modelling point-source signal-to-noise ratio for astronomical imaging systems.

It supports two operating modes:

- Analytic mode: build a diffraction-limited Airy PSF from first-order telescope parameters in a TOML file.
- Zemax mode on Windows: load a `.zmx` design through the ZOS-API and compute PSFs from OpticStudio.

The package produces SNR maps over exposure time and source magnitude, reports the SNR-optimal aperture, and marks regions where the brightest pixel saturates.

## Features

- Model telescope, sensor, bandpass, and sky background terms.
- Evaluate SNR on a 2D grid of exposure time and apparent magnitude.
- Optimize the photometric aperture from the PSF itself.
- Plot SNR contours and saturation boundaries.
- Use a built-in Sony IMX267 sensor model or load a sensor from TOML.
- Optionally drive PSF generation from Zemax OpticStudio on Windows.

## Installation

From the project root:

```bash
pip install .
```

From PyPI:

```bash
pip install astro-snr-calc
```

For editable development installs:

```bash
pip install -e .
```

The package requires Python 3.10 or newer.

## Command-Line Usage

After installation, the console entry point is:

```bash
SNRCalc <telescope.toml> [detector.toml] [options]
```

Current CLI options:

```text
usage: cli.py [-h] [-p PUPIL_SAMPLING] [-i IMAGE_SAMPLING]
              [-f [FIELD_INDEX ...]] [--no-plots] [--no-save]
              telescope [detector]
```

Examples:

```bash
SNRCalc telescope.toml
SNRCalc telescope.toml detector.toml
SNRCalc telescope.toml --no-plots
```

On Windows, you can also pass a Zemax design file instead of a telescope TOML file:

```bash
SNRCalc design.zmx detector.toml --field 1 3 --no-save
```

In analytic mode, the tool prints a model summary and saves an SNR map as a PNG file named after the telescope configuration stem.

## Configuration Files

### Telescope TOML

The telescope file describes the first-order optical model used in analytic mode.

```toml
name = "Example Telescope"
aperture = "80 mm"
effl = "400 mm"
wfno = 5.0
throughput = 0.85
```

Fields:

- `name`: descriptive telescope name.
- `aperture`: entrance aperture diameter.
- `effl`: effective focal length.
- `wfno`: working f-number.
- `throughput`: end-to-end transmission as a fraction from 0 to 1.

### Detector TOML

If you do not supply a detector file, the CLI uses a built-in Sony IMX267 uncooled model.

```toml
name = "Custom Sensor"
pixel_pitch = "3.45 um"
qe = 0.64
read_noise_e = 2.4
dark_rate = "3 1 / s"
full_well_e = 10700
```

Fields:

- `name`: descriptive detector name.
- `pixel_pitch`: detector pixel pitch.
- `qe`: quantum efficiency as a fraction from 0 to 1.
- `read_noise_e`: read noise in electrons RMS.
- `dark_rate`: dark current in electrons per second per pixel.
- `full_well_e`: full well capacity in electrons.

Unit-bearing values are parsed with `astropy.units.Quantity`, so values such as `"80 mm"`, `"3.45 um"`, and `"3 1 / s"` are accepted.

## Python API

You can also use the package directly from Python.

```python
import astropy.units as u

from snr_calc import (
    ExposureGrid,
    PhotometricBand,
    SNRResult,
    Sensor,
    SkyBrightness,
    Telescope,
)

sensor = Sensor(
    name="Sony IMX267 (uncooled)",
    pixel_pitch=3.45 * u.um,
    qe=0.64,
    read_noise_e=2.4,
    dark_rate=3.0 / u.s,
    full_well_e=10700.0,
)

telescope = Telescope(
    name="Example Telescope",
    aperture=80 * u.mm,
    effl=400 * u.mm,
    wfno=5.0,
    throughput=0.85,
)

model = telescope.build(
    sensor=sensor,
    band=PhotometricBand.johnson_v(),
    sky=SkyBrightness.dark_sky(),
)

result = SNRResult.from_model(model, *ExposureGrid.default().arrays())
print(model.summary())
```

Top-level exports include:

- `Sensor`
- `Telescope`
- `PhotometricBand`
- `SkyBrightness`
- `ExposureGrid`
- `DetectorPSF`
- `SNRModel`
- `SNRResult`
- `SNRMapPlotter`

## Zemax Support

Zemax integration is only available on Windows. The module raises an import error on other platforms.

When using a `.zmx` file, the CLI can request FFT or Huygens PSF calculations through OpticStudio. The Windows-only CLI also supports:

- `--huygens`: use Huygens PSF analysis instead of FFT.
- `--as-extension`: connect to a running interactive OpticStudio session.

If Zemax is not available, use analytic mode with a telescope TOML file.

## Output

For each evaluated model, the CLI:

- prints a textual summary of telescope, detector, sky, PSF, and aperture parameters,
- computes SNR over a default grid of 1 ms to 100 s and magnitude 0 to 16,
- saves a PNG SNR map unless `--no-save` is used,
- optionally displays the plot window unless `--no-plots` is used.

The plot includes logarithmic SNR coloring, contour overlays, and a hatched saturation boundary when the peak pixel exceeds the detector full well.

## Development

To run the CLI from the source tree without installing the package:

```bash
PYTHONPATH=src python -m snr_calc.cli --help
```