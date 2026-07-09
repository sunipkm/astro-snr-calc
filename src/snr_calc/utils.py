# %% Imports
from __future__ import annotations
import astropy.units as u
from typing import Annotated, TypeVar, get_type_hints, get_origin, get_args
import dataclasses
from typing import Any, List
from dacite import Config
from tomlkit import register_encoder
from tomlkit.items import Item as TomlItem, item as tomlitem
from astropy.units import Quantity
from numpy import ndarray, asarray, fromstring
from numpy.typing import NDArray


# %% Serde Helpers


@register_encoder
def qty_ndarray_encoder(obj: Any, /, _parent=None, _sort_keys=False) -> TomlItem:
    if isinstance(obj, Quantity):
        return tomlitem(f'{obj}')
    elif isinstance(obj, ndarray):
        return tomlitem(obj.tolist())
    else:
        raise TypeError(
            f'Object of type {type(obj)} is not JSON serializable.')


class QuantityDecoder:
    @staticmethod
    def decode_qty(value: str) -> Quantity:
        value = value.strip()
        if value.startswith('['):
            arr_str, unit = value.rsplit(']', 1)
            arr_str = arr_str.strip('[]')
            arr = fromstring(arr_str, sep=' ', dtype=float)
            return Quantity(arr, unit.strip())
        return Quantity(value)

    @staticmethod
    def decode_ndarray(value: List[Any]) -> NDArray:
        return asarray(value, dtype=float)

    @property
    def config(self) -> Config:
        return Config(
            type_hooks={
                Quantity: self.decode_qty,
                ndarray: self.decode_ndarray
            },
        )


QUANTITY_DECODER = QuantityDecoder().config


def _extract_quantity_ptype(hint):
    """Return the physical-type from a ``Quantity[ptype]`` annotation, or ``None``.

    ``u.Quantity['length']`` is syntactic sugar for
    ``typing.Annotated[Quantity, PhysicalType('length')]``.  We therefore look
    for ``Annotated`` as the outer wrapper, confirm the base type is
    ``Quantity``, and return the first metadata item (the ``PhysicalType``).
    """
    if get_origin(hint) is Annotated:
        args = get_args(hint)   # (base_type, *metadata)
        if args and args[0] is u.Quantity and len(args) > 1:
            return args[1]      # PhysicalType('length') etc.
    return None


_DC = TypeVar('_DC')


def validate_units(cls: type[_DC]) -> type[_DC]:
    """Class decorator that validates ``Quantity`` fields immediately after ``__init__``.

    Must be applied *after* ``@dataclass`` so that the generated ``__init__``
    is already present::

        @validate_units
        @dataclass(frozen=True)
        class MyClass:
            length: Quantity['length'] = field(metadata={'unit': u.mm})

    Two checks are performed for every field:

    1. **Physical-type annotation** - if a field is annotated as
       ``Quantity['<ptype>']`` (e.g. ``Quantity['length']``, ``Quantity['time']``),
       the value's physical type is verified against ``<ptype>`` using
       ``astropy.units.get_physical_type``.

    2. **Unit metadata** - if the ``dataclasses.field`` metadata contains a
       ``'unit'`` key, the value's unit must be equivalent to that unit
       (i.e. same physical dimensions, convertible via
       ``Unit.is_equivalent``).
    """
    if not dataclasses.is_dataclass(cls):
        raise TypeError(
            f"@validate_units must be applied to a dataclass, got {cls!r}"
        )

    orig_init = cls.__init__
    # Resolve annotations and field metadata once at decoration time.
    # include_extras=True preserves Annotated wrappers (stripped by default).
    try:
        hints = get_type_hints(cls, include_extras=True)
    except Exception:
        hints = dict(cls.__annotations__)
    field_map = {f.name: f for f in dataclasses.fields(cls)}

    def _validated_init(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        for name, hint in hints.items():
            if name not in field_map:
                continue
            value = getattr(self, name)
            fld = field_map[name]

            # --- Check 1: Quantity['physical_type'] annotation ---
            ptype_arg = _extract_quantity_ptype(hint)
            if ptype_arg is not None:
                if not isinstance(value, u.Quantity):
                    raise TypeError(
                        f"Field '{name}' annotated as Quantity[{ptype_arg!r}] "
                        f"but received {type(value).__name__!r}"
                    )
                expected_ptype = u.get_physical_type(ptype_arg)
                assert value.unit is not None
                actual_ptype = u.get_physical_type(
                    value.unit)  # type: ignore[arg-type]
                if actual_ptype != expected_ptype:
                    raise u.UnitsError(
                        f"Field '{name}': expected physical type "
                        f"{str(expected_ptype)!r}, "
                        f"got {str(actual_ptype)!r} (unit: {value.unit})"
                    )

            # --- Check 2: field metadata 'unit' ---
            meta_unit = fld.metadata.get('unit')
            if meta_unit is not None:
                if not isinstance(value, u.Quantity):
                    raise TypeError(
                        f"Field '{name}' requires a Quantity with unit equivalent "
                        f"to '{meta_unit}' but received {type(value).__name__!r}"
                    )
                # value.unit is typed Optional[UnitBase] in astropy's stubs but is
                # never None for a normally constructed Quantity; the check below
                # satisfies Pylance's type narrowing without hiding real bugs.
                if value.unit is None:  # pragma: no cover
                    raise TypeError(
                        f"Field '{name}': Quantity has no unit "
                        f"(expected unit equivalent to '{meta_unit}')"
                    )
                if not value.unit.is_equivalent(meta_unit):
                    raise u.UnitsError(
                        f"Field '{name}': unit '{value.unit}' is not equivalent "
                        f"to the required '{meta_unit}'"
                    )

    cls.__init__ = _validated_init
    return cls
