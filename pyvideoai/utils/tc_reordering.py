def get_TC_sort_idx(T: int):
    """TC reordering shuffles temporal ordering,
    so use this sorting indices to fix it
    """
    if T % 3 == 0:
        sort_idx = [(x*(T//3) + x//3) % T for x in range(T)]
    elif T % 3 == 1:
        sort_idx = [(x*((2*T+1)//3)) % T for x in range(T)]
    else:
        sort_idx = [(x*((T+1)//3)) % T for x in range(T)]
    return sort_idx
