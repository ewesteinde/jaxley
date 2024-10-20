from typing import Optional

from jaxley.pumps.pump import Pump


class CaPump(Pump):
    """Calcium dynamics tracking inside calcium concentration
    
    Modeled after Destexhe et al. 1994.
    """

    def __init__(self, name: Optional[str] = None):
        super().__init__(name)
        self.pump_params = {
            f"{self._name}_gamma": 0.05,  # Fraction of free calcium (not buffered).
            f"{self._name}_decay": 80,  # Buffering time constant in ms.
            f"{self._name}_depth": 0.1,  # Depth of shell in um.
            f"{self._name}_minCai": 1e-4,  # Minimum intracell. ca concentration in mM.
        }
        self.pump_states = {"CaCon_i": 5e-05}
        self.META = {
            "reference": "Modified from Destexhe et al., 1994",
            "mechanism": "Calcium dynamics",
        }

    def compute_current(self, u, dt, voltages, params):
        """Return change of calcium concentration based on calcium current and decay."""
        prefix = self._name
        ica = u["i_Ca"] / 1_000.0
        gamma = params[f"{prefix}_gamma"]
        decay = params[f"{prefix}_decay"]
        depth = params[f"{prefix}_depth"]
        minCai = params[f"{prefix}_minCai"]

        FARADAY = 96485  # Coulombs per mole.

        # Calculate the contribution of calcium currents to cai change.
        drive_channel = -10_000.0 * ica * gamma / (2 * FARADAY * depth)
        return drive_channel - (u["CaCon_i"] + minCai) / decay
