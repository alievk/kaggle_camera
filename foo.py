from tqdm import tqdm
from time import sleep

for j in range(3):
    tq = tqdm(total=100)
    tq.set_description('FOO')

    for i in range(100):
        tq.update(i)
        sleep(0.1)
