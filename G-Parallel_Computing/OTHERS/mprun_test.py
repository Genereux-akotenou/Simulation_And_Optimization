
"""
Function for memory test
"""
def f1(u, SIZE, np):
    u_array = np.array_split(u, SIZE)
    u_array_ = [np.concatenate(([u_array[i-1][-1]], u_array[i])) for i in range(1, len(u_array))]
    u_array_.insert(0, u_array[0])

def f2(u, SIZE, np):
    u_array = np.array_split(u, SIZE)
    for i in range(1, len(u_array)):
        u_array[i] = np.insert(u_array[i], 0, u_array[i-1][-1])
