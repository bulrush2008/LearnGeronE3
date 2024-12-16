
from zlib import crc32
import numpy as np

for i in range(10):
  hv = crc32(np.int64(i))
  print(hv)