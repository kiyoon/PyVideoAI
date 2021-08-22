def TC_idx(T):
    """
    Returns TC Reordering indices.
    When channels are ordered like (1R 1G 1B 2R 2G 2B 3R 3G 3B 4R ...),
    Returns indices to make it (1R 2R 3R 2G 3G 4G 3B 4B 5B 4R 5R 6R ...).
    """
    assert T >= 3
    idx_3frame = [0, 3, 6, 4, 7, 10, 8, 11, 14] # (1R 2R 3R) (2G 3G 4G) (3B 4B 5B)
    repeat = T//3 + (T%3>0) #math.ceil(T / 3)
    idx = []
    for i in range(repeat):
        idx.extend(map(lambda x: x+9*i, idx_3frame)) 
    idx = idx[:T*3] # We repeated too much so we truncate. Now, the index size is equal to T * 3 channels.
    def frame_limit(x):
        while x >= T*3:
            x -= 3
        return x
    idx = list(map(frame_limit, idx))
    return idx

def TCred_idx(T):
    """
    Returns TC Reordering indices, but using only red channel.
    When channels are ordered like (1R 1G 1B 2R 2G 2B 3R 3G 3B 4R ...),
    Returns indices to make it (1R 2R 3R 2R 3R 4R 3R 4R 5R 4R 5R 6R ...).
    """
    assert T >= 3
    idx_1frame = [0, 3, 6] # (1R 2R 3R)
    repeat = T
    idx = []
    for i in range(repeat):
        idx.extend(map(lambda x: x+3*i, idx_1frame)) 
    def frame_limit(x):
        while x >= T*3:
            x -= 3
        return x
    idx = list(map(frame_limit, idx))
    return idx

TC_idx_dict = {}
def TC_idx_fast(T, TC_idx_func=TC_idx):
    """Precompute TC indices"""
    if T not in TC_idx_dict.keys():
        TC_idx_dict[T] = TC_idx_func(T)
    return TC_idx_dict[T]

def NTCHW_to_TC_NTCHW(video_tensor, TC_idx_func=TC_idx):
    N, T, C, H, W = video_tensor.shape
    unrolled_video = video_tensor.reshape(N,-1,H,W)
    sort_idx = TC_idx_fast(T, TC_idx_func)
    tc_video = unrolled_video[:,sort_idx,...].reshape(N,T,C,H,W)
    return tc_video

def NCTHW_to_TC_NTCHW(video_tensor, TC_idx_func=TC_idx):
    return NTCHW_to_TC_NTCHW(video_tensor.permute(0,2,1,3,4), TC_idx_func)

def NCTHW_to_TC_NCTHW(video_tensor, TC_idx_func=TC_idx):
    return NCTHW_to_TC_NTCHW(video_tensor, TC_idx_func).permute(0,2,1,3,4)

# deprecated
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
