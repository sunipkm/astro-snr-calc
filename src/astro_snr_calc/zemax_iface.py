

# ======================================================================
# Generic class: ZemaxConfig            (instance: TWO_LENS_TELESCOPE_RUN)
#
# Everything that describes ONE analysis run of ONE Zemax design.  Passed
# explicitly into the classes that need it - no hidden global state, and
# several configs can coexist (e.g. to compare fields or sampling levels).
# ======================================================================
import sys

from .utils import validate_units

if sys.platform != "win32":
    raise ImportError("ZOS-API is only supported on Windows.")

from dataclasses import dataclass, field, replace
import logging
from pathlib import Path
from typing import Literal, Optional
import astropy.units as u
import numpy as np

from .snr_calculator import (
    DetectorPSF, PhotometricBand,
    Photometry, SNRModel, Sensor,
    SkyBrightness, Telescope,
)

PSFMethod = Literal["huygens", "fft"]
SampleSize = Literal[
    "S_32x32", "S_64x64", "S_128x128",
    "S_256x256", "S_512x512", "S_1024x1024",
    "S_2048x2048", "S_4096x4096", "S_8192x8192",
]

@validate_units
@dataclass(frozen=True)
class ZemaxConfig:
    """Any ZOS-API analysis run: design file + PSF analysis parameters."""

    zemax_file: Path              # path to the .zmx/.zos design
    # field point at which to evaluate the PSF.
    # This is 1-based, as in Zemax; the default of None means "all fields" and is handled by the SNRModel builder.
    field_index: Optional[int] = field(
        default=None, 
        metadata={
            'description': '1-based field index in the Zemax prescription; None means "all fields"'
            }
        )
    wave_index: int = 1          # wavelength number in the Zemax file
    image_delta: u.Quantity = 0.5 * u.um  # Huygens PSF image-plane sampling [um]
    throughput: float = 0.95     # bulk+coating transmission
    # PSF computation method:
    #   "huygens" - accurate for any system, but brute-force slow
    #   "fft"     - much faster; assumes a shift-invariant pupil and small
    #               field angles (fine for slow, near-axis systems)
    psf_method: PSFMethod = "fft"
    # Huygens cost ~ (pupil samples)^2 x (image samples)^2. Start small
    # ("S_32x32"/"S_64x64") to prove the pipeline, then increase.
    pupil_sampling: SampleSize = "S_32x32"
    image_sampling: SampleSize = "S_64x64"
    # Elapsed-time heartbeat while the analysis blocks (0 disables). The
    # ZOS-API provides no progress callback; for a true progress bar use
    # connect_as_extension=True and watch OpticStudio's own window.
    heartbeat_s: float = 5.0
    # If the hidden standalone instance hangs (invisible modal dialog),
    # open OpticStudio, enable Programming > Interactive Extension, and
    # set this True to connect to the visible instance instead:
    connect_as_extension: bool = False

    def __post_init__(self):
        filp = Path(self.zemax_file).absolute()
        if not filp.is_file():
            raise FileNotFoundError(
                f"Zemax file not found: {filp!r} - "
                f"the ZOS-API LoadFile would fail silently, so stopping here."
            )
        object.__setattr__(self, 'zemax_file', filp)

    def build(
        self,
        sensor: Sensor,
        band: PhotometricBand,
        sky: SkyBrightness,
        *,
        field_indices: list[int] = [],
    ) -> list[tuple[str, "SNRModel"]]:
        """Connect once, load once, then build one SNR model per field.

        field_indices:
            []          -> iterate over ALL fields defined in the prescription
            [i, j, ...] -> only these field numbers (1-based, as in Zemax)

        Returns a list of (field_label, SNRModel).  The connection, file load,
        and first-order extraction are done once; only the PSF analysis is
        re-run per field.
        """
        zos = ZOSConnection(connect_as_extension=self.connect_as_extension)
        models: list[tuple[str, SNRModel]] = []
        try:
            zos.load(str(self.zemax_file))
            logger.debug(
                "build: extracting first-order properties (EPD/EFFL/WFNO) ...")
            telescope = zos.build(
                throughput=self.throughput,
                wave_index=self.wave_index,
            )
            logger.debug(f"build: telescope = {telescope}")
            photometry = Photometry(telescope, sensor, band, sky)

            ftype, defined = zos.fields()
            by_index = {i: (x, y) for i, x, y in defined}
            indices = list(by_index) if not field_indices else field_indices
            unknown = [i for i in indices if i not in by_index]
            if unknown:
                raise ValueError(
                    f"Field index(es) {unknown} not in the prescription; "
                    f"defined fields are {sorted(by_index)}."
                )

            for i in indices:
                x, y = by_index[i]
                label = f"field {i} ({x:g}, {y:g}) [{ftype}]"
                logger.debug(f"build: computing PSF for {label} ...")
                psf = zos.to_psf(
                    pixel_size=sensor.pixel_pitch,
                    config=replace(self, field_index=i),
                )
                logger.debug(
                    f"build: {label}: peak fraction = {psf.peak_fraction:.3f}")
                models.append((label, SNRModel(photometry, psf)))
        finally:
            zos.close()
        return models


# Module-level logger - callers configure handlers/level via logging.getLogger()
logger = logging.getLogger(__name__)

# ======================================================================
# ZOS-API connection (standard boilerplate)
# ======================================================================


class ZOSConnection:
    """Standalone connection to Zemax OpticStudio through the ZOS-API."""

    def __init__(
        self,
        connect_as_extension: bool = False,
    ):
        import clr  # pythonnet
        import os
        import winreg

        logger.debug("ZOSConnection: reading Zemax root from registry ...")
        key = winreg.OpenKey(
            winreg.ConnectRegistry(None, winreg.HKEY_CURRENT_USER),
            r"Software\Zemax", 0, winreg.KEY_READ,
        )
        zemax_root = winreg.QueryValueEx(key, "ZemaxRoot")[0]
        winreg.CloseKey(key)
        logger.debug(f"ZOSConnection: ZemaxRoot = {zemax_root}")

        clr.AddReference(  # type: ignore
            os.path.join(zemax_root, r"ZOS-API\Libraries\ZOSAPI_NetHelper.dll")
        )
        import ZOSAPI_NetHelper  # type: ignore

        logger.debug("ZOSConnection: initializing ZOSAPI_NetHelper ...")
        if not ZOSAPI_NetHelper.ZOSAPI_Initializer.Initialize():
            raise RuntimeError("Could not initialize the ZOS-API.")
        zos_dir = ZOSAPI_NetHelper.ZOSAPI_Initializer.GetZemaxDirectory()
        logger.debug(f"ZOSConnection: OpticStudio directory = {zos_dir}")
        clr.AddReference(os.path.join(zos_dir, "ZOSAPI.dll"))  # type: ignore
        clr.AddReference(  # type: ignore
            os.path.join(zos_dir, "ZOSAPI_Interfaces.dll")
        )
        import ZOSAPI  # type: ignore

        self.ZOSAPI = ZOSAPI
        conn = ZOSAPI.ZOSAPI_Connection()
        if connect_as_extension:
            # Visible instance: lets you SEE dialogs/progress that would
            # otherwise hang a hidden standalone instance.
            logger.debug(
                "ZOSConnection: ConnectAsExtension(0) - OpticStudio must be "
                "open with Interactive Extension enabled ..."
            )
            self.app = conn.ConnectAsExtension(0)
        else:
            logger.debug(
                "ZOSConnection: CreateNewApplication() - starting a hidden "
                "OpticStudio instance (if this stalls, a modal dialog in "
                "the hidden instance is a likely cause; try "
                "connect_as_extension=True in the ZemaxConfig) ..."
            )
            self.app = conn.CreateNewApplication()
        if self.app is None:
            raise RuntimeError("Could not start OpticStudio.")
        logger.debug(
            f"ZOSConnection: connected, license valid for API = "
            f"{self.app.IsValidLicenseForAPI}"
        )
        if not self.app.IsValidLicenseForAPI:
            raise RuntimeError("License is not valid for ZOS-API use.")
        self.system = self.app.PrimarySystem
        logger.debug("ZOSConnection: PrimarySystem acquired")

    def load(self, path: str):
        import os
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Zemax file not found: {path!r} - LoadFile would fail "
                f"silently, so stopping here."
            )
        logger.debug(
            f"ZOSConnection: LoadFile({path}) "
            f"[{os.path.getsize(path)} bytes] ..."
        )
        ok = self.system.LoadFile(path, False)
        logger.debug(f"ZOSConnection: LoadFile returned {ok}")
        if not ok:
            raise RuntimeError(
                f"OpticStudio refused to load {path!r}. Common causes:\n"
                f"  * file was saved by a NEWER OpticStudio version than "
                f"the one the API launched\n"
                f"  * .zos archive not supported by this version's "
                f"LoadFile - open it in the GUI and re-save as .zmx\n"
                f"  * non-ASCII characters or cloud-placeholder path "
                f"(OneDrive/Dropbox file not synced locally)\n"
                f"Try opening the file manually in the same OpticStudio "
                f"version, File > Save As .zmx, and point the ZemaxConfig at "
                f"that."
            )
        lde = self.system.LDE
        logger.debug(
            f"ZOSConnection: loaded '{self.system.SystemName}' "
            f"(SystemFile = {self.system.SystemFile}), "
            f"{lde.NumberOfSurfaces} surfaces, "
            f"{self.system.SystemData.Fields.NumberOfFields} fields, "
            f"{self.system.SystemData.Wavelengths.NumberOfWavelengths} "
            f"wavelengths"
        )
        if lde.NumberOfSurfaces <= 3:
            raise RuntimeError(
                f"System has only {lde.NumberOfSurfaces} surfaces - this "
                f"looks like the default empty system, i.e. the file did "
                f"not actually load. Check the LoadFile diagnostics above."
            )

    def close(self):
        logger.debug("ZOSConnection: CloseApplication()")
        self.app.CloseApplication()

    def build(
        self,
        throughput: float,
        wave_index: int = 1,
    ) -> "Telescope":
        """Build a Telescope from the first-order properties of the loaded system.

        Args:
            throughput (float): bulk+coating transmission (dimensionless)
            wave_index (int, optional): wavelength index to use (1-based). Defaults to 1.

        Raises:
            RuntimeError: if the first-order values look like OpticStudio's empty default system.

        Returns:
            Telescope: the constructed Telescope object
        """
        sd = self.system.SystemData
        ap_type = sd.Aperture.ApertureType
        epd_enum = self.ZOSAPI.SystemData.ZemaxApertureType.EntrancePupilDiameter
        if ap_type == epd_enum:
            epd = sd.Aperture.ApertureValue
        else:
            epd = self.operand("EPDI")            # fallback: operand
        effl = self.operand("EFFL", 0, wave_index)
        wfno = self.operand("WFNO", 0, wave_index)
        if abs(effl) > 1e6 or wfno >= 1e4 or epd <= 0:
            raise RuntimeError(
                f"First-order values look like OpticStudio's empty default "
                f"system (EPD={epd}, EFFL={effl}, WFNO={wfno}) - the design "
                f"file is not actually loaded."
            )
        return Telescope(
            name=self.system.SystemName or "Zemax design",
            aperture=float(epd) * u.mm,
            effl=float(effl) * u.mm,
            wfno=float(wfno),
            throughput=throughput,
        )

    def to_psf(
        self,
        pixel_size: u.Quantity['length'],
        config: ZemaxConfig
    ) -> DetectorPSF:
        """Run a PSF analysis (Huygens or FFT, per `config`) and bin onto
        the detector pixel grid."""
        if pixel_size.unit.physical_type != 'length': # type: ignore
            raise ValueError(
                f"pixel_size must be a length quantity, got {pixel_size}"
            )
        ZOSAPI = self.ZOSAPI
        method = config.psf_method.lower()
        if method == "huygens":
            idm = ZOSAPI.Analysis.AnalysisIDM.HuygensPsf
        elif method == "fft":
            idm = ZOSAPI.Analysis.AnalysisIDM.FftPsf
        else:
            raise ValueError(
                f"Unknown psf_method {config.psf_method!r}; "
                f"use 'huygens' or 'fft'."
            )
        label = f"{method} PSF"
        logger.debug(f"from_zemax: creating {label} analysis ...")
        analysis = self.system.Analyses.New_Analysis(idm)
        # pythonnet 3.x returns the base IAS_ interface; downcast to reach
        # the analysis-specific settings members (Field, sampling, ...).
        st = self.downcast(analysis.GetSettings())
        logger.debug(
            f"from_zemax: settings type after downcast = "
            f"{type(st).__name__}"
        )
        st.Field.SetFieldNumber(config.field_index)
        st.Wavelength.SetWavelengthNumber(config.wave_index)

        if method == "huygens":
            st.PupilSampleSize = getattr(
                ZOSAPI.Analysis.SampleSizes,
                config.pupil_sampling,
            )
            st.ImageSampleSize = getattr(
                ZOSAPI.Analysis.SampleSizes,
                config.image_sampling,
            )
            st.ImageDelta = config.image_delta.to(u.um).value   # [um]
            # Read the settings BACK: if these do not match what was set,
            # the settings interface did not apply (version/casting issue)
            # and the analysis runs with defaults - possibly something much
            # heavier than intended.
            logger.debug(
                f"from_zemax: settings read-back - field="
                f"{st.Field.GetFieldNumber()}, "
                f"wave={st.Wavelength.GetWavelengthNumber()}, "
                f"pupil={st.PupilSampleSize}, image={st.ImageSampleSize}, "
                f"image delta={st.ImageDelta} um"
            )
            logger.debug(
                "from_zemax: Huygens PSF is brute-force; 100% CPU here is "
                "NORMAL. Cost scales ~ with (pupil samples)^2 x (image "
                "samples)^2; if it takes more than a few minutes, lower "
                "the sampling in the ZemaxConfig, or set "
                "psf_method='fft' ..."
            )
        else:
            # FFT PSF: pupil sampling only; image spacing is read from the
            # returned grid. Some settings are version-dependent, so apply
            # them guarded and log what stuck.
            samplesize = getattr(
                ZOSAPI.Analysis.Settings.Psf.PsfSampling,
                f'Psf{config.pupil_sampling}',
            )
            st.SampleSize = samplesize
            for attr, val in (
                (
                    "OutputSize",
                    getattr(
                        ZOSAPI.Analysis.SampleSizes,
                        config.image_sampling,
                    )
                ),
                ("ImageDelta", config.image_delta.to(u.um).value)
            ):
                try:
                    setattr(st, attr, val)
                    logger.debug(f"from_zemax: FFT setting {attr} = {val}")
                except Exception as exc:  # noqa: BLE001 - version-dependent
                    logger.debug(
                        f"from_zemax: FFT setting {attr} not available "
                        f"({exc}); using analysis default"
                    )
            logger.debug(
                f"from_zemax: settings read-back - field="
                f"{st.Field.GetFieldNumber()}, "
                f"wave={st.Wavelength.GetWavelengthNumber()}, "
                f"sample={st.SampleSize}"
            )
            logger.debug(
                "from_zemax: note - FFT PSF assumes a shift-invariant "
                "pupil; cross-check against Huygens once at low sampling."
            )

        DetectorPSF._apply_with_heartbeat(analysis, label, config.heartbeat_s)

        results = self.downcast(analysis.GetResults())
        if results is None or results.NumberOfDataGrids < 1:
            analysis.Close()
            raise RuntimeError(
                f"{label} returned no data grid - check the settings "
                f"read-back above and the analysis in the GUI."
            )
        grid = self.downcast(results.GetDataGrid(0))
        psf = self.net_to_numpy(grid.Values)
        dx_um = float(grid.Dx)                  # image-plane spacing [um]
        logger.debug(
            f"from_zemax: grid {psf.shape}, dx={dx_um} um, "
            f"sum={psf.sum():.4g}, peak={psf.max():.4g}"
        )
        if psf.sum() <= 0:
            raise RuntimeError(f"{label} grid is empty/zero.")
        analysis.Close()

        return DetectorPSF(
            DetectorPSF._bin_to_pixels(
                psf,
                dx_um,
                pixel_size.to(u.um).value,
            ),
            source=f"{label} (field {config.field_index})",
        )

    # ---- helpers ---------------------------------------------------------

    @staticmethod
    def downcast(obj):
        """Downcast a ZOS-API interface to its specific implementation.

        pythonnet 3.x returns objects typed as the *base* interface (e.g.
        IAS_ instead of IAS_HuygensPsf), hiding the analysis-specific
        members; `.__implementation__` recovers them.  pythonnet 2.5.2
        downcasts implicitly and has no such attribute, so fall through
        to the object itself there.
        """
        return getattr(obj, "__implementation__", obj)

    def fields(self) -> tuple[str, list[tuple[int, float, float]]]:
        """Enumerate the fields defined in the loaded prescription.

        Returns (field_type_name, [(index, X, Y), ...]).  X/Y are in the
        units implied by the field type (degrees for angle fields, lens
        units for object/image height fields).
        """
        fd = self.system.SystemData.Fields
        ftype = str(fd.GetFieldType())
        out = []
        for i in range(1, fd.NumberOfFields + 1):
            f = fd.GetField(i)
            out.append((i, float(f.X), float(f.Y)))
        logger.debug(f"ZOSConnection: field type = {ftype}, fields = {out}")
        return ftype, out

    def operand(self, name: str, *args) -> float:
        """Evaluate a merit-function operand, e.g. operand('EFFL', 0, 1)."""
        op = getattr(self.ZOSAPI.Editors.MFE.MeritOperandType, name)
        padded = (list(args) + [0] * 8)[:8]
        logger.debug(f"ZOSConnection: GetOperandValue({name}, {padded}) ...")
        val = self.system.MFE.GetOperandValue(op, *padded)
        logger.debug(f"ZOSConnection: {name} = {val}")
        return val

    @staticmethod
    def net_to_numpy(a) -> np.ndarray:
        """Convert a System.Double[,] to a numpy array.

        Element-by-element access across the .NET boundary is slow; for a
        256x256 grid this is ~65k interop calls and can take a while -
        progress is logged so it is not mistaken for a hang.
        """
        rows, cols = a.GetLength(0), a.GetLength(1)
        logger.debug(f"net_to_numpy: converting {rows}x{cols} .NET array ...")
        out = np.empty((rows, cols), dtype=float)
        for i in range(rows):
            for j in range(cols):
                out[i, j] = a[i, j]
            if logger.isEnabledFor(logging.DEBUG) and rows >= 128 and (i + 1) % 64 == 0:
                logger.debug(f"net_to_numpy: {i + 1}/{rows} rows")
        logger.debug("net_to_numpy: done")
        return out
