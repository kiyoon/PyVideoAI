def get_orderedTC_ordering(T: int):
    """TC swap shuffles temporal ordering,
    so use this sorting indices to fix it
    """
    if T % 3 == 0:
        ordering = [(x*(T//3) + x//3) % T for x in range(T)]
    elif T % 3 == 1:
        ordering = [(x*((2*T+1)//3)) % T for x in range(T)]
    else:
        ordering = [(x*((T+1)//3)) % T for x in range(T)]
    return ordering
