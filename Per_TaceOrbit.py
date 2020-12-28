"""Test to run a tracking simulation"""
import Lib as ss
import numpy as np

Marker = {'R': np.array([1.2, 1.8, 2.2, 1.4], dtype=np.float64),
          'z': np.array([0.08, 0.08, 0.08, 0.08], dtype=np.float64),
          'phi': 1.0 * np.array([0.08, 0.08, 0.08, 0.08], dtype=np.float64),
          'vR': 0.0 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
          'vt': 1e6 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
          'vz': 0.0 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
          'm': 2.0 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
          'q': 1.0 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
          'logw': 0.0 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64),
          't': 0.0 * np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float64)}

B = ss.ihibpsim.prepare_B_field(32312, 0.35)

E = B.copy()
E['fz'] = E['fz'] * 0.0
E['ft'] = E['ft'] * 0.0
E['fr'] = E['fr'] * 0.0

ss.ihibpsim.write_fields('B.pod', B)
ss.ihibpsim.write_fields('E.pod', E)
ss.ihibpsim.write_markers('M.pod', Marker)
