"""Generate random block matrices."""
from argparse import ArgumentParser
from random import sample, shuffle
import numpy as np
from tqdm.auto import tqdm

parser = ArgumentParser()
parser.add_argument("-n", type=int, required=True, help="Matrix size. n for an nxn matrix.")
parser.add_argument("-m", type=int, required=True, help="Block size. m for an mxm block.")
parser.add_argument("-z", type=int, dest="non_zero", required=True, help="#Non-zero blocks.")
parser.add_argument("-o", dest="output", type=str, required=True, help="Output file name.")
args = parser.parse_args()
print(args)

n = args.n
m = args.m
k = args.non_zero
assert n % m == 0, "n mod(m) != 0"
assert k <= (n // m)**2, \
    f"#Non-zero blocks supplied ({k}) is more than the maximum possible number ({(n // m)**2})."

# Choose coordinates for non-overlapping blocks.
coordinates = [(row // m, col // m) for col in range(0, n, m) for row in range(0, n, m)]

print('Sampling coordinates...')
shuffle(coordinates)
non_zero_coordinates = coordinates[:k]
print('Sampled coordinates')

# choose whether to place a non zero block or not.
with open(args.output, "wb") as file:
    file.write(n.to_bytes(length=4, byteorder="little"))
    file.write(m.to_bytes(length=4, byteorder="little"))
    file.write(k.to_bytes(length=4, byteorder="little"))
    for row, col in tqdm(non_zero_coordinates):
        block = np.random.uniform(low=0, high=2**10, size=(m, m)).astype("<u2")
        # block = np.full((m,m), 2**16-1, dtype='<u2')
        file.write(row.to_bytes(length=4, byteorder="little"))
        file.write(col.to_bytes(length=4, byteorder="little"))
        file.write(block.tobytes())
