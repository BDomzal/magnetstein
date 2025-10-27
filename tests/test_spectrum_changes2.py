import numpy as np
import sys
import masserstein
from masserstein.spectrum import Spectrum
from masserstein.nmr_spectrum import NMRSpectrum

class CustomSpectrum(Spectrum):
    pass

def test_add_returns_same_class():
    s1 = CustomSpectrum(confs=[(1.0, 2.0)])
    s2 = CustomSpectrum(confs=[(2.0, 3.0)])
    result = s1 + s2
    assert isinstance(result, CustomSpectrum)

def test_add_returns_nmrspectrum():
    s1 = NMRSpectrum(confs=[(1.0, 2.0)])
    s2 = NMRSpectrum(confs=[(2.0, 3.0)])
    result = s1 + s2
    assert isinstance(result, NMRSpectrum)

def test_add_returns_spectrum():
    s1 = Spectrum(confs=[(1.0, 2.0)])
    s2 = Spectrum(confs=[(2.0, 3.0)])
    result = s1 + s2
    assert isinstance(result, Spectrum)


def test_mul_returns_same_class():
    s = CustomSpectrum(confs=[(1.0, 2.0)])
    result = s * 2
    assert isinstance(result, CustomSpectrum)

def test_scalar_product_returns_same_class():
    s1 = CustomSpectrum(confs=[(1.0, 2.0)])
    s2 = CustomSpectrum(confs=[(1.0, 3.0)])
    result = CustomSpectrum.ScalarProduct([s1, s2], [0.5, 0.5])
    assert isinstance(result, CustomSpectrum)

def test_resample_returns_same_class():
    s = CustomSpectrum(confs=[(1.0, 2.0), (2.0, 3.0)])
    target_mz = np.linspace(1.0, 2.0, 5)
    result = s.resample(target_mz)
    assert isinstance(result, CustomSpectrum)

def test_filter_against_other_returns_same_class():
    s1 = CustomSpectrum(confs=[(1.0, 2.0), (2.0, 3.0)])
    s2 = CustomSpectrum(confs=[(1.0, 2.0)])
    result = s1.filter_against_other([s2], margin=0.5)
    assert isinstance(result, CustomSpectrum)

def test_new_from_fasta(tmp_path):
    fasta_content = ">test\nACDE\n"
    fasta_file = tmp_path / "test.fasta"
    fasta_file.write_text(fasta_content)
    s = Spectrum.new_from_fasta(str(fasta_file))
    assert isinstance(s, Spectrum)
    assert hasattr(s, "confs")

def test_new_from_csv(tmp_path):
    csv_content = "mz,intensity\n100,1.0\n101,0.5\n"
    csv_file = tmp_path / "test.csv"
    csv_file.write_text(csv_content)
    s = Spectrum.new_from_csv(str(csv_file))
    assert isinstance(s, Spectrum)
    assert hasattr(s, "confs")
    assert len(s.confs) == 2

def test_new_random():
    s = Spectrum.new_random(peaks=5, domain=(100, 200))
    assert isinstance(s, Spectrum)
    assert hasattr(s, "confs")
    assert len(s.confs) == 5

def test_fuzzify_peaks_modifies_confs():
    s = Spectrum(confs=[(100.0, 1.0), (101.0, 0.5)])
    original_confs = list(s.confs)
    s.fuzzify_peaks(sd=0.1, step=0.01)
    # After fuzzification, number of confs should increase or change
    assert hasattr(s, "confs")
    assert len(s.confs) >= len(original_confs)
    # At least one m/z value should be different
    assert any(abs(c[0] - o[0]) > 1e-6 for c, o in zip(s.confs, original_confs))