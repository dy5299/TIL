import numpy as np


def tabl(n, m):
    f = np.zeros((n, m))
    f[0, :] = [i + 1 for i in range(m)]
    f[:, 0] = 1
    for i in range(1, n):
        for j in range(1, m):
            f[i, j] = int(sum(f[i - 1][0:j + 1]))

    return f  # np.sort(f,axis=0)[::-1]


print(tabl(14, 13))
