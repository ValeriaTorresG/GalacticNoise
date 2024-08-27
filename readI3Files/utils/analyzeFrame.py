# class AnalyzeQFrames(icetray.I3Module):
#     def __init__(self, ctx):  
#         icetray.I3Module.__init__(self, ctx)
        
#         self.NFreqBins = int(args.traceBins/ 2 + 1)
        
#     def getSpectrumInfo(self,frame,keyName):
#         radio_hits = frame[keyName]
#         for num_, (AntennaKey, pulses) in enumerate(radio_hits):
#             pol1_pulse, pol2_pulse = pulses[0] , pulses[1]
#             # Polarization 1
#             pol1_pulseDict = spectrum.spectrum_extractor(pol1_pulse, args.getTimeSeries, args.getFFT, args.getEnvelope)
#             # Polarization 2
#             pol2_pulseDict = spectrum.spectrum_extractor(pol2_pulse, args.getTimeSeries, args.getFFT, args.getEnvelop)
#             assert len(pol1_pulseDict["hilbertPy"]) == len(pol2_pulseDict["hilbertPy"]) , f"Length of pulses should be same. Got {len(pol1_pulse), len(pol2_pulse)}"
#             if len(pol1_pulseDict["hilbertPy"]) <= 0: continue # Skip the empty pulses        
            
