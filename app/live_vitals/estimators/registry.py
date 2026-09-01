"""Registry of available vital-sign estimators.

The app selects estimators by name so that adding a model, or comparing two on the
same data, is a configuration change rather than a code change. Nothing outside this
module should import a concrete estimator class directly.
"""
from .hr_physnet import HRPhysNet
from .hr_spectral import HRSpectral

ESTIMATORS = {
    HRPhysNet.name: HRPhysNet,
    HRSpectral.name: HRSpectral,
}
DEFAULT_ESTIMATOR = HRPhysNet.name


def available():
    """Returns the sorted names of all registered estimators."""
    return sorted(ESTIMATORS)


def available_for(vital):
    """Returns the names of registered estimators producing the given vital."""
    return sorted(name for name, cls in ESTIMATORS.items() if cls.vital == vital)


def get_estimator(name=None, **kwargs):
    """Instantiates a registered estimator by name."""
    name = name or DEFAULT_ESTIMATOR
    if name not in ESTIMATORS:
        raise KeyError(f"unknown estimator {name!r}; available: {available()}")
    return ESTIMATORS[name](**kwargs)
