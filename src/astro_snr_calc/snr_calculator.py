# %%
from dataclasses import dataclass, field
import logging
import time
from typing import Optional

from serde_dataclass import TomlDataclass, toml_config
from .utils import QUANTITY_DECODER, validate_units

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import astropy.units as u

logger = logging.getLogger(__name__)

# %%


# ======================================================================
# Sensor
# ======================================================================
@validate_units
@dataclass(frozen=True)
@toml_config(de=QUANTITY_DECODER)
class Sensor(TomlDataclass):
    """Any image sensor described by its noise and geometry parameters."""

    name: str
    pixel_pitch: u.Quantity['length']           # pixel pitch
    # quantum efficiency (dimensionless)
    qe: float
    # read noise [e- rms] - electrons dimensionless
    read_noise_e: float
    # dark current [e-/s/px] - electrons dimensionless → 1/s
    dark_rate: u.Quantity['frequency']
    # full-well capacity [e-] - electrons dimensionless
    full_well_e: float


# ======================================================================
# PhotometricBand
# ======================================================================
@validate_units
@dataclass(frozen=True)
@toml_config(de=QUANTITY_DECODER)
class PhotometricBand(TomlDataclass):
    """Any photometric band: reference wavelength + photon zero point."""

    name: str
    wavelength: u.Quantity['length']    # reference wavelength
    zero_point: u.Quantity = field(metadata={'unit': 1 / (u.s * u.cm * u.cm)})
    # photon flux for m=0 source [ph/s/cm^2]; photons are dimensionless → 1/(s·cm^2)

    @staticmethod
    def johnson_v() -> "PhotometricBand":
        """Return the Johnson V band."""
        return PhotometricBand(
            name="Johnson V",
            wavelength=0.55 * u.um,
            zero_point=1000.0 * 880.0 / (u.s * u.cm**2),
        )

    @staticmethod
    def johnson_b() -> "PhotometricBand":
        """Return the Johnson B band."""
        return PhotometricBand(
            name="Johnson B",
            wavelength=0.44 * u.um,
            zero_point=1400.0 * 980.0 / (u.s * u.cm**2),
        )

# ======================================================================
# SkyBrightness
# ======================================================================


@dataclass(frozen=True)
class SkyBrightness(TomlDataclass):
    """Any sky, described by its surface brightness [mag/arcsec^2].

    Kept as a plain float because magnitudes are used logarithmically and
    astropy's magnitude-unit system would add complexity without benefit here.
    """

    name: str = field(metadata={'description': "sky description"})
    mag: float = field(
        metadata={'description': "surface brightness [mag/arcsec^2]"})

    @staticmethod
    def dark_sky() -> "SkyBrightness":
        """Return a typical dark rural sky."""
        return SkyBrightness(name="dark rural sky", mag=21.8)

    @staticmethod
    def suburban_sky() -> "SkyBrightness":
        """Return a typical suburban sky."""
        return SkyBrightness(name="suburban sky", mag=20.5)

    @staticmethod
    def urban_sky() -> "SkyBrightness":
        """Return a typical urban sky."""
        return SkyBrightness(name="urban sky", mag=18.0)


# ======================================================================
# ExposureGrid
# ======================================================================
@validate_units
@dataclass(frozen=True)
@toml_config(de=QUANTITY_DECODER)
class ExposureGrid(TomlDataclass):
    """Any (exposure time x magnitude) evaluation grid."""

    t_min: u.Quantity['time']
    t_max: u.Quantity['time']
    mag_min: float
    mag_max: float
    n_t: int = 400
    n_mag: int = 400

    def arrays(self):
        """Return ``(t_exp, mags)``; ``t_exp`` is a Quantity in seconds."""
        t = np.logspace(
            np.log10(self.t_min.to(u.s).value),
            np.log10(self.t_max.to(u.s).value),
            self.n_t,
        ) * u.s
        m = np.linspace(self.mag_min, self.mag_max, self.n_mag)
        return t, m

    @staticmethod
    def default() -> "ExposureGrid":
        """Return the default exposure grid."""
        return ExposureGrid(
            t_min=1e-3 * u.s,
            t_max=100.0 * u.s,
            mag_min=0.0,
            mag_max=16.0,
        )

# ======================================================================
# Generic class: Telescope  (instance: built from your .zos file)
# ======================================================================


@validate_units
@dataclass(frozen=True)
@toml_config(de=QUANTITY_DECODER)
class Telescope(TomlDataclass):
    """First-order properties of a telescope, pulled from OpticStudio."""

    name: str
    aperture: u.Quantity['length'] = field(metadata={'unit': u.mm})
    effl: u.Quantity['length'] = field(metadata={'unit': u.mm})
    wfno: float                    # working F-number (dimensionless)
    throughput: float              # bulk+coating transmission (dimensionless)

    def build(
        self,
        sensor: Sensor,
        band: PhotometricBand,
        sky: SkyBrightness
    ) -> "SNRModel":
        """No Zemax: analytic diffraction-limited PSF through the same pipeline."""
        psf = DetectorPSF.from_airy(
            wfno=self.wfno,
            wavelength_um=band.wavelength.to(u.um).value,
            pixel_um=sensor.pixel_pitch.to(u.um).value,
        )
        return SNRModel(Photometry(self, sensor, band, sky), psf)

    @property
    def collecting_area(self) -> u.Quantity:
        """Unobstructed collecting area."""
        r = (self.aperture / 2).to(u.cm)
        return np.pi * r ** 2

    def plate_scale(self, pixel: u.Quantity) -> u.Quantity:
        """Angular size of one pixel, returned in arcsec (per pixel implicit)."""
        return u.Quantity((pixel / self.effl).decompose() * u.rad).to(u.arcsec)


# ======================================================================
# Generic class: DetectorPSF
#   (instance: from the Huygens PSF, or an analytic Airy in demo mode)
# ======================================================================
@dataclass(frozen=True)
class DetectorPSF:
    """A point-spread function binned onto the detector pixel grid.

    `pixel_fractions` holds, per pixel, the fraction of the total source
    flux falling on that pixel (sums to <= 1; energy outside the simulated
    window is simply lost, which is conservative).
    """

    pixel_fractions: np.ndarray
    source: str = "unspecified"

    # ---- constructors ------------------------------------------------------
    @staticmethod
    def _apply_with_heartbeat(analysis, label: str, interval_s: float):
        """Run ApplyAndWaitForCompletion with elapsed-time heartbeats.

        The ZOS-API has no progress callback, so this cannot show a
        percentage - but pythonnet releases the GIL during the blocking
        .NET call, letting a background thread print elapsed time so a
        long computation is distinguishable from a hang. (For Zemax's own
        native progress bar, connect with connect_as_extension=True and
        watch the OpticStudio window.)
        """
        import threading

        t_start = time.perf_counter()
        stop = threading.Event()

        def beat():
            while not stop.wait(interval_s):
                logger.debug(
                    f"{label}: still computing ... "
                    f"{time.perf_counter() - t_start:.0f} s elapsed"
                )

        beater = None
        if interval_s > 0:
            beater = threading.Thread(target=beat, daemon=True)
            beater.start()
        logger.debug(f"{label}: ApplyAndWaitForCompletion() ...")
        try:
            analysis.ApplyAndWaitForCompletion()
        finally:
            stop.set()
            if beater is not None:
                beater.join(timeout=1.0)
        logger.debug(
            f"{label}: finished in {time.perf_counter() - t_start:.1f} s")

    @classmethod
    def from_airy(
        cls,
        wfno: float,
        wavelength_um: float,
        pixel_um: float,
        sample_um: float = 0.05,
        window_um: float = 60.0
    ) -> "DetectorPSF":
        """Analytic diffraction-limited Airy PSF (demo / cross-check mode)."""
        from scipy.special import j1

        n = int(round(window_um / sample_um)) | 1        # odd sample count
        x = (np.arange(n) - n // 2) * sample_um
        X, Y = np.meshgrid(x, x)
        r = np.hypot(X, Y)
        u = np.pi * r / (wavelength_um * wfno)
        with np.errstate(divide="ignore", invalid="ignore"):
            psf = np.where(u == 0.0, 1.0, (2.0 * j1(u) / u) ** 2)
        return cls(
            cls._bin_to_pixels(psf, sample_um, pixel_um),
            source="analytic Airy",
        )

    # ---- internals -----------------------------------------------------------
    @staticmethod
    def _bin_to_pixels(
        psf: np.ndarray,
        dx_um: float,
        pixel_um: float,
    ) -> np.ndarray:
        """Sum fine PSF samples into detector pixels (star at pixel center)."""
        total = psf.sum()
        samples_per_pix = max(1, int(round(pixel_um / dx_um)))
        ny, nx = psf.shape
        # crop symmetrically so the PSF peak sits at the center of a pixel
        npx_y = ny // samples_per_pix
        npx_x = nx // samples_per_pix
        cy = (ny - npx_y * samples_per_pix) // 2
        cx = (nx - npx_x * samples_per_pix) // 2
        cropped = psf[cy:cy + npx_y * samples_per_pix,
                      cx:cx + npx_x * samples_per_pix]
        binned = cropped.reshape(
            npx_y, samples_per_pix, npx_x, samples_per_pix
        ).sum(axis=(1, 3))
        return binned / total

    # ---- derived quantities -----------------------------------------------------
    @property
    def sorted_fractions(self) -> np.ndarray:
        """Per-pixel flux fractions, brightest first."""
        return np.sort(self.pixel_fractions.ravel())[::-1]

    @property
    def peak_fraction(self) -> float:
        """Fraction of the source flux on the brightest pixel."""
        return float(self.sorted_fractions[0])

    def best_aperture(
        self,
        source_rate_ref: float,
        bkg_rate_per_pix: float,
        read_noise_e: float,
        t_ref: float,
    ) -> tuple[int, float]:
        """SNR-optimal aperture: (n_pix, enclosed-energy fraction).

        Scans apertures made of the k brightest pixels and picks the k that
        maximizes SNR for a reference source rate and exposure time (the
        classic enclosed-energy vs. background-pixels trade-off).
        """
        f = self.sorted_fractions
        ee = np.cumsum(f)
        k = np.arange(1, f.size + 1)
        sig = ee * source_rate_ref * t_ref
        var = sig + k * bkg_rate_per_pix * t_ref + k * read_noise_e ** 2
        snr = sig / np.sqrt(var)
        i = int(np.argmax(snr))
        return int(k[i]), float(ee[i])


# ======================================================================
# Generic class: Photometry
# ======================================================================
@dataclass(frozen=True)
class Photometry:
    """Any photometric setup: optics, detector, band, and sky."""

    telescope: Telescope
    sensor: Sensor
    band: PhotometricBand
    sky: SkyBrightness

    def _detected_rate(self, mag) -> u.Quantity:
        """Detected rate [1/s] for a source at apparent magnitude `mag`.

        When `mag` is a surface brightness in mag/arcsec^2 the result is
        the rate per arcsec^2; divide by arcsec^2 to make that explicit.
        """
        return (
            self.band.zero_point                       # 1/(s·cm^2)
            * 10.0 ** (-0.4 * np.asarray(mag, dtype=float))
            * self.telescope.collecting_area           # cm^2
            * self.telescope.throughput
            * self.sensor.qe
        )                                              # -> 1/s

    def source_rate(self, mag) -> u.Quantity:
        """Total point-source rate arriving at the focal plane [1/s]."""
        return self._detected_rate(mag)

    @property
    def plate_scale(self) -> u.Quantity:
        """Angular size of one pixel [arcsec]."""
        return self.telescope.plate_scale(self.sensor.pixel_pitch)

    @property
    def sky_rate_per_pixel(self) -> u.Quantity:
        """Sky background rate [1/s] per pixel."""
        rate_per_arcsec2 = self._detected_rate(self.sky.mag) / u.arcsec**2
        return u.Quantity(rate_per_arcsec2 * self.plate_scale ** 2).to(1/u.s)


# ======================================================================
# Generic class: SNRModel (PSF-aware)
# ======================================================================
@dataclass(frozen=True)
class SNRResult:
    t_grid: u.Quantity          # exposure time meshgrid [s]
    m_grid: np.ndarray          # magnitude meshgrid (dimensionless)
    snr: np.ndarray             # SNR (dimensionless)
    saturated: np.ndarray       # bool mask: brightest pixel exceeds full well
    _model: "SNRModel" = field(init=False, repr=False, compare=False)

    @classmethod
    def from_model(cls, model: "SNRModel", t_exp: u.Quantity, mags: np.ndarray) -> "SNRResult":
        """Evaluate the SNR model on a grid of exposure times and magnitudes."""
        obj = model._evaluate(t_exp, mags)
        object.__setattr__(obj, "_model", model)
        return obj


@dataclass(frozen=True)
class SNRModel:
    """SNR equation using aperture terms derived from the real PSF."""

    photometry: Photometry
    psf: DetectorPSF
    ref_mag: float = 12.0        # reference source for aperture optimization
    ref_t_s: float = 1.0

    @property
    def telescope(self) -> Telescope:
        return self.photometry.telescope

    @property
    def sensor(self) -> Sensor:
        return self.photometry.sensor

    def aperture(self) -> tuple[int, float]:
        """(n_pix, enclosed-energy fraction) of the SNR-optimal aperture."""
        p, s = self.photometry, self.sensor
        return self.psf.best_aperture(
            source_rate_ref=p.source_rate(self.ref_mag).to(1/u.s).value,
            bkg_rate_per_pix=u.Quantity(
                p.sky_rate_per_pixel + s.dark_rate).to(1/u.s).value,
            read_noise_e=s.read_noise_e,
            t_ref=self.ref_t_s,
        )

    def _evaluate(self, t_exp: u.Quantity, mags: np.ndarray) -> SNRResult:
        T_val, M = np.meshgrid(t_exp.to(u.s).value, mags)
        T = T_val * u.s
        p, s = self.photometry, self.sensor

        n_pix, ee = self.aperture()
        S = p.source_rate(M)                       # Quantity [1/s], 2-D
        B = p.sky_rate_per_pixel                   # Quantity [1/s], scalar

        signal = (ee * S * T).decompose().value
        noise = np.sqrt(
            signal
            + n_pix * ((B + s.dark_rate) * T).decompose().value
            + n_pix * s.read_noise_e ** 2
        )
        snr = signal / noise

        peak_e = ((self.psf.peak_fraction * S + B +
                  s.dark_rate) * T).decompose().value
        saturated = peak_e >= s.full_well_e
        return SNRResult(T, M, snr, saturated)

    def summary(self) -> str:
        p, t, s = self.photometry, self.telescope, self.sensor
        n_pix, ee = self.aperture()
        lines = [
            f"Telescope                : {t.name}",
            f"  D / EFFL / F#          : {t.aperture.to(u.mm).value:.3f} mm / "
            f"{t.effl.to(u.mm).value:.3f} mm / {t.wfno:.4f}",
            f"Sensor                   : {s.name}",
            f"Band / sky               : {p.band.name} / {p.sky.name} "
            f"({p.sky.mag} mag/arcsec^2)",
            f"PSF source               : {self.psf.source}",
            f"Peak-pixel fraction      : {self.psf.peak_fraction:.3f}",
            f"Optimal aperture         : {n_pix} px "
            f"(enclosed energy {ee:.3f}, optimized at m={self.ref_mag}, "
            f"t={self.ref_t_s} s)",
            f"Plate scale              : {p.plate_scale.to(u.arcsec).value:.2f} arcsec/px",
            f"Sky background per pixel : {p.sky_rate_per_pixel.to(1/u.s).value:.3f} e-/s/px",
            f"m=10 source rate         : {p.source_rate(10.0).to(1/u.s).value:.1f} e-/s",
        ]
        return "\n".join(lines)


# ======================================================================
# Generic class: SNRMapPlotter
# ======================================================================
@dataclass(frozen=True)
class SNRMapPlotter:
    cmap: str = "viridis"
    contour_levels: tuple = (1, 3, 5, 10, 30, 100, 300)
    vmin: float = 1e-2

    def plot(self, result: SNRResult, savepath: Optional[str] = None):
        if not isinstance(result, SNRResult):
            raise TypeError(f"Expected SNRResult, got {type(result).__name__}")
        if result._model is None:
            raise ValueError(
                "SNRResult must be created via SNRModel.from_model()")
        fig, ax = plt.subplots(figsize=(9, 6.5))
        # plain ndarray - matplotlib can't handle Quantity
        t_s = result.t_grid.to(u.s).value
        pcm = ax.pcolormesh(
            t_s, result.m_grid, result.snr,
            norm=LogNorm(vmin=self.vmin, vmax=result.snr.max()),
            cmap=self.cmap, shading="auto",
        )
        fig.colorbar(pcm, ax=ax, label="SNR")
        cs = ax.contour(
            t_s, result.m_grid, result.snr,
            levels=list(self.contour_levels), colors="white", linewidths=0.9,
        )
        ax.clabel(cs, fmt="SNR=%g", fontsize=8, colors="white")

        if result.saturated.any():
            ax.contourf(t_s, result.m_grid, result.saturated,
                        levels=[0.5, 1.5], colors="none", hatches=["////"])
            ax.contour(
                t_s, result.m_grid, result.saturated,
                levels=[0.5], colors="red", linewidths=1.2
            )
            ax.plot([], [], color="red", label="pixel saturation limit")
            ax.legend(loc="lower right", framealpha=0.8)

        t, s, p = result._model.telescope, result._model.sensor, result._model.photometry
        n_pix, ee = result._model.aperture()
        ax.set_xscale("log")
        ax.set_xlabel("Exposure time [s]")
        ax.set_ylabel(f"Apparent magnitude ({p.band.name})")
        ax.invert_yaxis()
        ax.set_title(
            f"Point-source SNR - D={t.aperture.to(u.mm).value:.1f} mm, "
            f"EFFL={t.effl.to(u.mm).value:.3f} mm, F/{t.wfno:.4f} - {result._model.psf.source}\n"
            f"{s.name}: QE={s.qe:.2f}, RN={s.read_noise_e} e-, "
            f"dark={s.dark_rate.to(1/u.s).value} e-/s/px, {n_pix}-px aperture (EE={ee:.2f}), "
            f"{p.plate_scale:.1f}\"/px"
        )
        fig.tight_layout()
        if savepath:
            fig.savefig(savepath, dpi=150)
        return fig, ax
