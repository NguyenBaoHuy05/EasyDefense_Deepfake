import csv

class MetricLogger:
    def __init__(self, filename):
        self.filename = filename
        self.metrics = {}

    def collect(self, name, value):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def save(self):
        with open(self.filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            headers = list(self.metrics.keys())
            writer.writerow(headers)
            rows = zip(*self.metrics.values())
            writer.writerows(rows)