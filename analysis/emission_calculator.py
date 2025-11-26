
class EmissionCalculator:
    CAR_EMISSION = 20     # g/min idle
    BIKE_EMISSION = 8     # g/min idle

    def calculate(self, count):
        # Approx. mix: 70% bikes, 30% cars
        bikes = int(count * 0.7)
        cars = count - bikes

        total = (bikes * self.BIKE_EMISSION) + (cars * self.CAR_EMISSION)
        return total
