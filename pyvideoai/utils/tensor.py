
# https://stackoverflow.com/questions/63960352/reshaping-order-in-pytorch-fortran-like-index-ordering
def reshape_fortran(x, shape):
    if len(x.shape) > 0:
        x = x.permute(*reversed(range(len(x.shape))))
    return x.reshape(*reversed(shape)).permute(*reversed(range(len(shape))))
