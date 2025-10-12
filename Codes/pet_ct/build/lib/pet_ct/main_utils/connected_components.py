
import cc3d
import numpy as np

def cc(input, connectivity=26, dust=False, threshold_dust=10, return_N = False):
    if return_N:
        output, N = cc3d.connected_components(np.array(input), connectivity=connectivity, out_dtype=np.uint16, return_N=return_N)
    else:
        output = cc3d.connected_components(np.array(input), connectivity=connectivity, out_dtype=np.uint16)

    if dust:
        output = cc3d.dust(
            output, threshold=threshold_dust,
            connectivity=connectivity, in_place=False
        )

    if return_N:
        return output, N
    else:
        return output
