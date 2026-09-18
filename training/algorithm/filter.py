import numpy as np
class LowPassFilter:
    def __init__(self, cutoff_freq=50, dt=0.004):
        """
        一阶低通滤波器
        cutoff_freq: 截止频率 (Hz)
        dt: 采样时间间隔 (s)
        """
        self.dt = dt
        self.cutoff_freq = cutoff_freq
        self.filtered_value = None
    
    def filter(self, new_value, dt=None):
        if dt is None: dt = self.dt
        self.alpha = 2 * np.pi * dt * self.cutoff_freq / (2 * np.pi * dt * self.cutoff_freq + 1)
        if self.filtered_value is None:
            self.filtered_value = new_value.copy()
        else:
            self.filtered_value = self.alpha * new_value + (1 - self.alpha) * self.filtered_value
        return self.filtered_value
    
    def reset(self):
        self.filtered_value = None
