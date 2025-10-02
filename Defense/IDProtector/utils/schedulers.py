class WarmupScheduler:
    def __init__(self, optimizer, warmup_steps, initial_lr, lr, current_step = 0):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.initial_lr = initial_lr
        self.lr = lr
        self.current_step = current_step

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.initial_lr + (self.lr - self.initial_lr) * self.current_step / self.warmup_steps
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

class LinearInterpolator:
    def __init__(self, t_min, t_max):
        self.t_min = t_min
        self.t_max = t_max

    def __call__(self, a, b, t):
        if t <= self.t_min:
            return a
        elif t >= self.t_max:
            return b
        else:
            return a + (b - a) * (t - self.t_min) / (self.t_max - self.t_min)

class MultiStepLinearInterpolator:
    def __init__(self, times):
        self.times = times

    def __call__(self, values, t):
        if len(values) != len(self.times):
            raise ValueError("Length of values must match length of time steps.")
        
        if t <= self.times[0]:
            return values[0]
        elif t >= self.times[-1]:
            return values[-1]
        
        for i in range(1, len(self.times)):
            if self.times[i-1] <= t <= self.times[i]:
                t0, t1 = self.times[i-1], self.times[i]
                v0, v1 = values[i-1], values[i]
                return v0 + (v1 - v0) * (t - t0) / (t1 - t0)
