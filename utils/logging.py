"""
Logging utilities.
"""
from collections import defaultdict, deque
import datetime
import functools
import time

import torch

from .distributed import is_dist_avail_and_initialized


class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        self.deque = deque(maxlen=window_size)
        self.count = 0
        self.total = 0.0
        self.fmt = "{avg:.4f}" if fmt is None else fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    def synchronize_between_processes(self):
        """
        Synchronizes the count and total attributes across processes.

        Note that it does not synchronize the deque.
        """

        if not is_dist_avail_and_initialized():
            return

        t = torch.tensor([self.count, self.total], dtype=torch.float64, device='cuda')
        torch.distributed.barrier()
        torch.distributed.all_reduce(t)
        t = t.tolist()
        self.count = int(t[0])
        self.total = t[1]

    @property
    def median(self):
        return torch.tensor(list(self.deque)).median().item()

    @property
    def avg(self):
        return sum(self.deque) / len(self.deque)

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value)


class MetricLogger(object):
    def __init__(self, delimiter="\t", window_size=20):
        self.delimiter = delimiter
        self.meters = defaultdict(functools.partial(SmoothedValue, window_size=window_size))

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __str__(self):
        str_list = [f"{name}: {meter}" for name, meter in self.meters.items()]
        logger_str = self.delimiter.join(str_list)

        return logger_str

    def synchronize_between_processes(self):
        for meter in self.meters.values():
            meter.synchronize_between_processes()

    def log_every(self, iterable, print_freq, header=""):
        data_time = SmoothedValue(fmt="{avg:.4f}")
        iter_time = SmoothedValue(fmt="{avg:.4f}")

        space_fmt = ':' + str(len(str(len(iterable)))) + 'd'
        if torch.cuda.is_available():
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'eta: {eta}',
                'data_time: {data_time}',
                'iter_time: {iter_time}',
                'mem: {mem:.0f}'
            ])
        else:
            log_msg = self.delimiter.join([
                header,
                '[{0' + space_fmt + '}/{1}]',
                '{meters}',
                'eta: {eta}',
                'data_time: {data_time}',
                'iter_time: {iter_time}'
            ])

        start_time = time.time()
        pre_iter_time = time.time()

        for i, obj in enumerate(iterable, 1):
            data_time.update(time.time() - pre_iter_time)
            yield obj
            iter_time.update(time.time() - pre_iter_time)

            if i % print_freq == 0 or i == len(iterable):
                eta = iter_time.global_avg * (len(iterable) - i)
                eta = str(datetime.timedelta(seconds=int(eta)))
                max_memory = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)

                print(log_msg.format(i, len(iterable), eta=eta, meters=str(self), iter_time=str(iter_time),
                      data_time=str(data_time), mem=max_memory))

            pre_iter_time = time.time()

        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        print('{} Total time: {} ({:.4f} s / it)'.format(header, total_time_str, total_time / len(iterable)))
