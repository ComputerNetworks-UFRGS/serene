import random
import time


class StragglerGenerator:
    def __init__(self, pattern):
        self.pattern = pattern
    
    def __repr__(self):
        return self.pattern.__class__.__name__

    def step(self, base_time):
        print(f"Testing if should straggle")
        if self.pattern.should_straggle():
            sleep_duration = self.pattern.straggling_duration(base_time)
            print(f"Straggling for {sleep_duration:.4f}s")
            time.sleep(sleep_duration)
            return sleep_duration
        return 0

    @classmethod
    def getInstance(cls, pattern_name, **kwargs):
        patterns = {
            'slow_worker': SlowWorkerPattern,
            'failure': FailurePattern,
            'heterogeneous': HeterogeneousPattern,
        }
        return cls(patterns[pattern_name](**kwargs))

class SlowWorkerPattern:
    def __init__(self, probability, min_slowdown, max_slowdown, **kwargs):
        self.probability = probability
        self.min_slowdown = min_slowdown
        self.max_slowdown = max_slowdown

    def should_straggle(self):
        return random.random() <= self.probability
    
    def straggling_duration(self, base_time):
        straggling_time = random.uniform(self.min_slowdown, self.max_slowdown)
        print("Straggling time:", straggling_time, base_time)
        return straggling_time * base_time


class FailurePattern:
    def __init__(self, probability, failure_duration, **kwargs):
        self.probability = probability
        self.failure_duration = failure_duration
        self.has_failed = False
    
    def should_straggle(self):
        if self.has_failed:
            return False
        self.has_failed = random.random() <= self.probability
        return self.has_failed

    def straggling_duration(self, base_time):
        return self.failure_duration


class HeterogeneousPattern:
    def __init__(self, constant_slowdown, **kwargs):
        self.slow_down = constant_slowdown

    def should_straggle(self):
        return True

    def straggling_duration(self, base_time):
        return self.slow_down * base_time
