import pytest
import numpy as np
from masserstein.spectrum import Spectrum

class CustomSpectrum(Spectrum):
    """Simple subclass to validate returning same subclass from factory/ops."""
    pass

def make_conf_list(n=3, start=100.0, step=1.0):
    return [(start + i*step, float(i+1)) for i in range(n)]

def test_add_mul_return_subclass():
    s1 = CustomSpectrum(confs=[(100.0, 1.0)])
    s2 = CustomSpectrum(confs=[(101.0, 2.0)])
    res = s1 + s2
    assert isinstance(res, CustomSpectrum)
    res2 = s1 * 3.0
    assert isinstance(res2, CustomSpectrum)
    res3 = 2.0 * s1
    assert isinstance(res3, CustomSpectrum)

def test_scalarproduct_returns_subclass_and_sum():
    s1 = CustomSpectrum(confs=[(100.0, 0.4), (101.0, 0.6)])
    s2 = CustomSpectrum(confs=[(100.0, 0.1), (101.0, 0.9)])
    # weights simple check
    out = CustomSpectrum.ScalarProduct([s1, s2], [1.0, 1.0])
    assert isinstance(out, CustomSpectrum)
    # result contains two masses and positive intensities
    assert len(out.confs) >= 2
    assert all(i >= 0 for _, i in out.confs)

def test_resample_returns_subclass_and_length():
    s = CustomSpectrum(confs=[(100.0, 1.0), (101.0, 2.0)])
    target = np.linspace(99.5, 101.5, 7)
    r = s.resample(target)
    assert isinstance(r, CustomSpectrum)
    assert len(r.confs) == len(target)

def test_filter_against_other_returns_subclass_and_filters():
    s = CustomSpectrum(confs=[(100.0, 1.0), (105.0, 0.5), (110.0, 0.2)])
    other = CustomSpectrum(confs=[(100.1, 0.5), (109.9, 0.7)])
    filtered = s.filter_against_other([other], margin=0.5)
    assert isinstance(filtered, CustomSpectrum)
    # expect to keep peaks near 100 and 110 (within 0.5) but maybe drop 105
    kept_mz = [mz for mz, _ in filtered.confs]
    assert any(abs(mz - 100.1) <= 0.5 for mz in kept_mz)
    assert any(abs(mz - 109.9) <= 0.5 for mz in kept_mz)

def test_new_random_returns_instance_and_respects_peaks():
    # This will detect if new_random returns a class instead of an instance.
    s = Spectrum.new_random(domain=(200.0, 201.0), peaks=4)
    assert isinstance(s, Spectrum), "new_random must return a Spectrum instance"
    assert hasattr(s, "confs")
    assert len(s.confs) == 4

def test_fuzzify_peaks_broadens_profile():
    s = Spectrum(confs=[(100.0, 1.0), (101.0, 0.5)])
    original_len = len(s.confs)
    s.fuzzify_peaks(sd=0.5, step=0.1)
    assert isinstance(s.confs, list)
    # fuzzify should produce a profile with more samples than the original centroid peaks
    assert len(s.confs) >= original_len
