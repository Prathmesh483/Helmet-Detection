# analysis/emission_calculator.py
EMISSION_FACTORS_G_PER_SEC = {
    "car": 2.3,         # grams CO2 per second idling (example)
    "motorbike": 0.6,
    "bus": 6.5,
    "truck": 8.0,
    "bicycle": 0.0
}

class EmissionCalculator:
    def __init__(self):
        pass

    def calculate(self, counts, idle_seconds=60):
        """
        counts: integer or dict per vehicle type
        if counts is a single integer, treat as cars
        returns grams CO2 for idle_seconds
        """
        if isinstance(counts, int):
            counts = {"car": counts}
        total = 0.0
        breakdown = {}
        for k, v in counts.items():
            ef = EMISSION_FACTORS_G_PER_SEC.get(k, EMISSION_FACTORS_G_PER_SEC["car"])
            g = ef * v * idle_seconds
            breakdown[k] = g
            total += g
        return {"total_gCO2": total, "breakdown": breakdown}
