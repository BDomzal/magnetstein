from masserstein import Spectrum

class NMRSpectrum(Spectrum):
    def __init__(self, formula='', threshold=0.001, total_prob=None,
                 charge=1, adduct=None, confs=None, label=None, protons=None, **other):
        
        Spectrum.__init__(self, formula=formula, threshold=threshold, total_prob=total_prob,
                 charge=charge, adduct=adduct, confs=confs, label=label, **other)
        self.protons = protons