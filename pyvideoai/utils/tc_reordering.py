def get_frame_idx_1D(frame_idx, colour='R'):
    """
    frame_idx (int): starts from one

    example:
        get_frame_idx_1D(1, 'R') == 0
        get_frame_idx_1D(2, 'R') == 3
        get_frame_idx_1D(2, 'G') == 4
    """
    assert colour in ['R', 'G', 'B']
    assert frame_idx > 0
    # first calculate index for R
    R_idx = (frame_idx-1) * 3
    if colour == 'R':
        return R_idx
    elif colour == 'G':
        return R_idx + 1
    elif colour == 'B':
        return R_idx + 2

def get_frame_indices_1D(list_frame_colour):
    """
    example:
        get_frame_indices_1D(['1R', '2R', '3R']) == [0, 3, 6]
    """
    def get_frame_idx_from_str(str_frame_colour):
        return get_frame_idx_1D(int(str_frame_colour[:-1]), str_frame_colour[-1])

    return list(map(get_frame_idx_from_str, list_frame_colour))



def TC_idx(T):
    """
    Returns TC Reordering indices.
    When channels are ordered like (1R 1G 1B 2R 2G 2B 3R 3G 3B 4R ...),
    Returns indices to make it (1R 2R 3R 2G 3G 4G 3B 4B 5B 4R 5R 6R ...).
    """
    assert T >= 3
    #idx_3frame = [0, 3, 6, 4, 7, 10, 8, 11, 14] # (1R 2R 3R) (2G 3G 4G) (3B 4B 5B)
    idx_3frame = get_frame_indices_1D(['1R', '2R', '3R', '2G', '3G', '4G', '3B', '4B', '5B'])
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

def TCrgb_idx(T):
    """
    Returns TC Reordering indices, but alternating R G B.
    When channels are ordered like (1R 1G 1B 2R 2G 2B 3R 3G 3B 4R ...),
    Returns indices to make it (1R 2G 3B 2R 3G 4B 3R 4G 5B 4R 5G 6B ...).
    """
    assert T >= 3
    idx_1frame = get_frame_indices_1D(['1R', '2G', '3B'])
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

def TCred_idx(T):
    """
    Returns TC Reordering indices, but using only red channel.
    When channels are ordered like (1R 1G 1B 2R 2G 2B 3R 3G 3B 4R ...),
    Returns indices to make it (1R 2R 3R 2R 3R 4R 3R 4R 5R 4R 5R 6R ...).
    """
    assert T >= 3
    idx_1frame = get_frame_indices_1D(['1R', '2R', '3R'])
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

def TCShortLong_idx(T):
    """
    TC Short-Long
    Returns TC Reordering indices, but including long-term as well as short-term sampling.
    Idea is instead of duplicating channels, use long-term sampling.
    When channels are ordered like (1R 1G 1B 2R 2G 2B 3R 3G 3B 4R ...),
    Returns indices to make it (1R 2R 3R 2G 3G 4G 3B 4B 5B 4R 5R 6R 5G 6G 7G 6B 7B 8B '3R 5R 7R 4G 6G 8G'...).
    """
    assert T == 8, 'only supporting 8-frame'
    idx = TC_idx(T)
    idx[-6:] = get_frame_indices_1D(['3R', '5R', '7R', '4G', '6G', '8G'])
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
