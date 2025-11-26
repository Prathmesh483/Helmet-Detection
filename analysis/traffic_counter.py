# analysis/traffic_counter.py
class TrafficCounter:
    def __init__(self):
        pass

    def count(self, vehicle_list):
        """
        vehicle_list: list of dicts each with 'label'
        returns integer total or dict breakdown
        """
        if not isinstance(vehicle_list, list):
            return 0
        breakdown = {}
        for v in vehicle_list:
            lbl = v.get("label", "unknown")
            breakdown[lbl] = breakdown.get(lbl, 0) + 1
        return breakdown
