from tqdm import tqdm
# from tqdm.auto import tqdm  # notebook compatible
import time
for i1 in tqdm(range(10),desc = 'OVER RUN'):
    for i2 in tqdm(range(100), leave=False, desc='sub run'):
        # do something, e.g. sleep
        time.sleep(0.01)