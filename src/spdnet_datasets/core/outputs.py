"""Output contract of the datasets (see ADR-0002).

By default a sample is its covariance. Optional outputs are appended in a
fixed order::

    cov | (cov, mean) | (cov, raw) | (cov, mean, raw)

Datasets that only provide raw data or channel means (most SAR datasets) declare
``outputs=("raw", "mean")``; they return ``raw`` (``return_raw=True``) or
``mean`` (``return_mean=True``), exactly one of the two.
"""

ALL_OUTPUTS = ("cov", "mean", "raw")


def validate_output_request(outputs=ALL_OUTPUTS, return_mean=False, return_raw=False):
    """Raise ``ValueError`` if the requested outputs are not available."""
    if "cov" not in outputs:
        if return_mean == return_raw:
            raise ValueError(
                f"This dataset provides {tuple(outputs)} only: set exactly one of "
                "return_raw=True or return_mean=True."
            )
    if return_mean and "mean" not in outputs:
        raise ValueError(f"return_mean=True is not available; outputs are {tuple(outputs)}")
    if return_raw and "raw" not in outputs:
        raise ValueError(f"return_raw=True is not available; outputs are {tuple(outputs)}")


def pack_outputs(cov=None, mean=None, raw=None, *, return_mean=False, return_raw=False,
                 outputs=ALL_OUTPUTS):
    """Assemble one sample's input according to the output contract."""
    validate_output_request(outputs, return_mean, return_raw)
    if "cov" not in outputs:
        return raw if return_raw else mean
    parts = [cov]
    if return_mean:
        parts.append(mean)
    if return_raw:
        parts.append(raw)
    return parts[0] if len(parts) == 1 else tuple(parts)
