from icecube import radcube
from icecube.dataclasses import fft
from icecube.icetray import I3Units
import numpy as np

def spectrum_extractor(signalPulse, getTimeSeries=False, getFFT=False, getEnvelope = False):
    pulse_ = signalPulse
    fftData = pulse_.GetFFTData()
    if getTimeSeries:
        times_ind, tsPy = radcube.RadTraceToPythonList(fftData.GetTimeSeries())
    else : times_ind, tsPy = None, None
    
    if getFFT:
        freqs, spectrumPy = radcube.RadTraceToPythonList(fftData.GetFrequencySpectrum())
        # spectrumPy = np.abs(spectrumPy)
        specBinning = fftData.GetFrequencySpectrum().binning
        amplitude = [radcube.GetDbmHzFromFourierAmplitude(abs(thisAmp), specBinning,
                                                          50 * I3Units.ohm) for thisAmp in spectrumPy]
    else: freqs, spectrumPy, specBinning,amplitude = None, None, None, None
    
    # Hilbert Envelope
    if getEnvelope:
        hilbert = fft.GetHilbertEnvelope(fftData)
        times, hilbertPy = radcube.RadTraceToPythonList(hilbert)
    else : times, hilbertPy = None, None
    
    # Hilbert Peak
    hilbertPeak = fft.GetHilbertPeak(fftData)
    hilbertPeakTime = fft.GetHilbertPeakTime(fftData)
    
    outDict = dict(times_ind = times_ind, tsPy = tsPy, 
                   freqs = freqs, spectrumPy = spectrumPy , specBinning=specBinning,
                   times = times, hilbertPy = hilbertPy ,
                   hilbertPeak = hilbertPeak, hilbertPeakTime = hilbertPeakTime,
                   amplitude = amplitude)
    
    return outDict