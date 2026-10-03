#METRICS (drift)
#PSI (population stability index) is the primary metric used to detect drift in the input distribution -> PSI = sum_bins (actual_pct - expected_pct) * ln(actual_pct / expected_pct)
#in other words, remember first 100 prompt lengths (reference window), bucket them into 5 bins by size, then compare the percentages of prompts in each bin between the reference and current windows
#it measures the relative change in the distribution of a feature (prompt length) between a reference window and a current window allowing us to spot irregularities/significant shifts in the distribution of prompt sizes
#generally, a PSI value below 0.1 indicates no significant shift, 0.1-0.2 indicates a moderate shift, and above 0.2 indicates a significant shift in the distribution of prompt lengths
import math
from collections import deque

from prometheus_client.core import GaugeMetricFamily

from config import block_size, drift_reference_size, drift_window_size

_NUM_BINS = 5 #number of bins for splitting up prompt lengths (bin 1: 0-20, bin 2: 20-40, etc.)
_EPS = 1e-4 #small number added to percentages so an empty bin never causes log(0) or a divide by zero

#bin creation: divides a list of values into a fixed number of bins based on the bin width
def _bin_proportions(values, bin_width):
    counts = [0] * _NUM_BINS #initializes bin counts to zero
    for v in values: #for each prompt length v in the list of values
        idx = min(int(v / bin_width), _NUM_BINS - 1) #
        counts[idx] += 1
    total = len(values)
    return [c / total for c in counts]

#DriftMonitor tracks prompt length distribution over time and computes PSI
class DriftMonitor:
    def __init__(self, reference_size=drift_reference_size, window_size=drift_window_size):
        self.reference_size = reference_size
        self.reference = [] #first reference_size prompt lengths seen, frozen once full
        self.window = deque(maxlen=window_size) #most recent window_size prompt lengths
        self._bin_width = block_size / _NUM_BINS #max prompt length / number of bins

    #observe a new prompt length, updating both the reference and current windows as needed
    def observe(self, prompt_length):
        if len(self.reference) < self.reference_size: #append to reference window while it's not full
            self.reference.append(prompt_length)
        self.window.append(prompt_length) #append to the current window after reference is full

    #psi computation
    def psi(self):
        #none until there's enough data on both sides for the comparison to mean anything
        if len(self.reference) < self.reference_size or len(self.window) < self.window.maxlen:
            return None

        expected = _bin_proportions(self.reference, self._bin_width)
        actual = _bin_proportions(self.window, self._bin_width)

        score = 0.0
        for e, a in zip(expected, actual):
            e, a = e + _EPS, a + _EPS
            score += (a - e) * math.log(a / e)
        return score

#DriftCollector hands the PSI score off to Prometheus
class DriftCollector:
    #kept separate from DriftMonitor so the PSI math stays a plain statistics
    #object, testable without touching a metrics registry
    def __init__(self, monitor):
        self.monitor = monitor

    def collect(self):
        #the current PSI reading
        gauge = GaugeMetricFamily(
            "llmis_prompt_length_psi",
            "Population Stability Index of recent prompt-length distribution vs the reference window (<0.1 stable, 0.1-0.2 moderate shift, >=0.2 significant shift).",
        )
        score = self.monitor.psi()
        if score is not None:
            gauge.add_metric([], score)
        yield gauge
