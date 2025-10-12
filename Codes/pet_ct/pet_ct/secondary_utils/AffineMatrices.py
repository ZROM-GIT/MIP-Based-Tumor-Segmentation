from munch import Munch

affine_matrices = Munch({'SLA': [[0, 1, 0, 0],
                                       [0, 0, -1, 0],
                                       [-1, 0, 0, 0],
                                       [0, 0, 0, 1]]
                         ,
                              'SRA': [[0, -1, 0, 0],
                                       [0, 0, -1, 0],
                                       [-1, 0, 0, 0],
                                       [0, 0, 0, 1]]
                         })