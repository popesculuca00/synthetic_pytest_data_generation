import torch

def timeseries_interpolate_batch_times(times, values, t):
  
  gi = torch.remainder(torch.sum((times - t) <= 0,dim = 0), times.shape[0])
  y2 = torch.diagonal(values[gi])
  y1 = torch.diagonal(values[gi-1])
  t2 = torch.diagonal(times[gi])
  t1 = torch.diagonal(times[gi-1])

  slopes = (y2 - y1) / (t2 - t1)
  return y1 + slopes * (t - t1)