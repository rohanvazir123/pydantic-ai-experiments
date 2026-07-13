
'''
Moving Average: Design a data structure that 
calculates the moving average of a streaming telemetry metric over the last N seconds under high throughput.
'''


import time

class TelemetryRollingAverage:
    def __init__(self, max_age_seconds: float):
        self.buffer = []
        self.max_age = max_age_seconds
        
        # Keep track of the running totals in memory
        self.running_sum = 0.0
        self.running_count = 0

    def add_batch(self, sorted_batch: list):
        """Adds a sorted batch of telemetry data points."""
        if not sorted_batch:
            # No new data to add, just evict expired elements
            self.evict_expired()
            return

        # 1. Add new points to running stats BEFORE merging
        new_sum = sum(point['value'] for point in sorted_batch)
        self.running_sum += new_sum
        self.running_count += len(sorted_batch)

        # 2. Merge the sorted batch using Timsort (O(N))
        self.buffer.extend(sorted_batch)
        self.buffer.sort(key=lambda x: x['timestamp'])

        # 3. Evict expired elements and subtract them from running stats
        self.evict_expired()

    def evict_expired(self):
        cutoff_time = time.time() - self.max_age
        
        # Track what needs to be removed
        expired_count = 0
        expired_sum = 0.0

        # Scan from the left (oldest) to find expired elements
        for point in self.buffer:
            if point['timestamp'] < cutoff_time:
                expired_sum += point['value']
                expired_count += 1
            else:
                break  # Stop immediately at the first valid point

        if expired_count > 0:
            # Subtract the expired values from our running totals
            self.running_sum -= expired_sum
            self.running_count -= expired_count
            
            # Slice the buffer to drop them from memory
            self.buffer = self.buffer[expired_count:]

    def get_moving_average(self) -> float:
        """Returns the current moving average in O(1) constant time."""

        # Evict expired elements first to ensure the average is correct
        self.evict_expired()

        if self.running_count == 0:
            return 0.0
        return self.running_sum / self.running_count


