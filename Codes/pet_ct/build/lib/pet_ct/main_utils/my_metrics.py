"""
This file contains self-made metrics.
"""
import monai
import cc3d
import numpy as np


def sensitivity(y_pred, y):
    confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y, include_background=False)
    value = monai.metrics.compute_confusion_matrix_metric("sensitivity", confusion_matrix)
    return value


def specificity(y_pred, y):
    confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y, include_background=False)
    value = monai.metrics.compute_confusion_matrix_metric("specificity", confusion_matrix)
    return value


def precision(y_pred, y):
    confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y, include_background=False)
    value = monai.metrics.compute_confusion_matrix_metric("precision", confusion_matrix)
    return value


def negative(y_pred, y):
    confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y, include_background=False)
    value = monai.metrics.compute_confusion_matrix_metric("negative", confusion_matrix)
    return value


def accuracy(y_pred, y):
    confusion_matrix = monai.metrics.get_confusion_matrix(y_pred=y_pred, y=y, include_background=False)
    value = monai.metrics.compute_confusion_matrix_metric("accuracy", confusion_matrix)
    return value


# class Detection:
#     def __init__(self,
#                  batch_size=1,
#                  spatial_dims=3,
#                  bounding_box_dims=2,
#                  spatial_dim_to_iter=0
#                  ):
#
#         self.batch_size = batch_size
#         self.spatial_dims = spatial_dims
#         self.bouding_box_dims = bounding_box_dims
#         self.spatial_dim_to_iter = spatial_dim_to_iter
#
#     def __call__(self, y_pred, y):
#         # Make sure dimensions match
#         self._check_dimensions(y_pred=y_pred, y=y)
#
#         # Argmax one-hot tensor
#         y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)
#
#         # Main loop for calculating the metric
#         self.main_loop(y_pred=y_pred, y=y)
#
#     def main_loop(self, y_pred, y):
#         for b in range(self.batch_size):
#             self._create_bounding_boxes(y_pred=y_pred[b], y=y[b])
#
#     def _check_dimensions(self, y_pred, y):
#         if (len(y_pred.shape) != self.spatial_dims + 2):
#             raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
#         elif (len(y.shape) != self.spatial_dims + 2):
#             raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')
#
#     def _argmax_and_squeeze(self, y_pred, y):
#         y_pred = np.argmax(y_pred, axis=1)
#         y = np.argmax(y, axis=1)
#         return y_pred, y
#
#     def _create_bounding_boxes(self, y_pred, y):
#         y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
#         if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
#             Precision = []
#             Recall = []
#             F1 = []
#
#             for m in range(y_pred.shape[0]):
#                 TP = 0
#
#                 labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
#                 labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)
#
#                 stats = cc3d.statistics(labels)
#                 stats_pred = cc3d.statistics(labels_pred)
#
#                 for i in range(1, N_pred + 1):
#                     x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, stats_pred['bounding_boxes'][i][0].stop
#                     y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, stats_pred['bounding_boxes'][i][1].stop
#                     found = False
#                     j = 1
#                     while (not found) & (j <= N):
#                         x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
#                         y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop
#
#                         if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
#                             TP += 1
#                             found = True
#                         j += 1
#                 FP = N_pred - TP
#                 TP_rev = 0
#
#                 for i in range(1, N + 1):
#                     x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
#                     y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
#                     found = False
#                     j = 1
#                     while (not found) & (j <= N_pred):
#                         x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, stats_pred['bounding_boxes'][j][0].stop
#                         y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, stats_pred['bounding_boxes'][j][1].stop
#
#                         if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
#                             TP_rev += 1
#                             found = True
#                         j += 1
#
#                 FN = N - TP_rev
#
#                 # Calculate complex metrics
#                 Precision_m = TP / (TP + FP)
#                 Recall_m = TP / (TP + FN)
#                 F1_m = 2 * ((Precision_m * Recall_m) / (Precision_m + Recall_m))
#
#                 # TP_total += TP
#                 # FP_total += FP
#                 # FN_total += FN
#                 # N_total += N
#                 # N_pred_total += N_pred
#
#                 Precision.append(Precision_m)
#                 Recall.append(Recall_m)
#                 F1.append(F1_m)
#
#             # Calculate complex metrics
#             # Precision_total = TP_total / (TP_total + FP_total)
#             # Recall_total = TP_total / (TP_total + FN_total)
#             # F1_total = 2 * ((Precision_total * Recall_total) / (Precision_total + Recall_total))
#
#             Precision = np.mean(Precision)
#             Recall = np.mean(Recall)
#             F1 = np.mean(F1)
#
#
#         elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
#             pass
#
#         return y_pred, y

class Precision_all:
    def __init__(self,
                 batch_size=1,
                 spatial_dims=3,
                 bounding_box_dims=2,
                 spatial_dim_to_iter=0,
                 mode='mean'):

        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.bouding_box_dims = bounding_box_dims
        self.spatial_dim_to_iter = spatial_dim_to_iter
        self.mode = mode

    def __call__(self, y_pred, y):
        # Make sure dimensions match
        self._check_dimensions(y_pred=y_pred, y=y)

        # Argmax one-hot tensor
        y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)

        # Main loop for calculating the metric
        Precision = self.main_loop(y_pred=y_pred, y=y)

        return Precision

    def main_loop(self, y_pred, y):
        Precision = []
        for b in range(self.batch_size):
            Precision.append(self._create_bounding_boxes(y_pred=y_pred[b], y=y[b]))
        return Precision

    def _check_dimensions(self, y_pred, y):
        if (len(y_pred.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
        elif (len(y.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')

    def _argmax_and_squeeze(self, y_pred, y):
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return y_pred, y

    def _create_bounding_boxes(self, y_pred, y):
        y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
        if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
            TP_total = 0
            FP_total = 0
            FN_total = 0
            N_total = 0
            N_pred_total = 0

            for m in range(y_pred.shape[0]):
                TP = 0

                labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
                labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)

                N_total += N
                N_pred_total += N_pred

                stats = cc3d.statistics(labels)
                stats_pred = cc3d.statistics(labels_pred)

                for i in range(1, N_pred + 1):
                    x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, \
                    stats_pred['bounding_boxes'][i][0].stop
                    y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, \
                    stats_pred['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N):
                        x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
                        y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP += 1
                            found = True
                        j += 1
                FP = N_pred - TP
                TP_rev = 0

                for i in range(1, N + 1):
                    x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
                    y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N_pred):
                        x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, \
                        stats_pred['bounding_boxes'][j][0].stop
                        y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, \
                        stats_pred['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP_rev += 1
                            found = True
                        j += 1

                FN = N - TP_rev

                TP_total += TP
                FP_total += FP
                FN_total += FN

            if (N_pred_total == 0) & (N_total != 0):
                return 0

            # Calculate complex metrics
            Precision = TP_total / (TP_total + FP_total)

        elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
            pass

        return Precision


class Recall_all:
    def __init__(self,
                 batch_size=1,
                 spatial_dims=3,
                 bounding_box_dims=2,
                 spatial_dim_to_iter=0,
                 mode='mean'):

        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.bouding_box_dims = bounding_box_dims
        self.spatial_dim_to_iter = spatial_dim_to_iter
        self.mode = mode

    def __call__(self, y_pred, y):
        # Make sure dimensions match
        self._check_dimensions(y_pred=y_pred, y=y)

        # Argmax one-hot tensor
        y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)

        # Main loop for calculating the metric
        Recall = self.main_loop(y_pred=y_pred, y=y)

        return Recall

    def main_loop(self, y_pred, y):
        Recall = []
        for b in range(self.batch_size):
            Recall.append(self._create_bounding_boxes(y_pred=y_pred[b], y=y[b]))
        return Recall

    def _check_dimensions(self, y_pred, y):
        if (len(y_pred.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
        elif (len(y.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')

    def _argmax_and_squeeze(self, y_pred, y):
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return y_pred, y

    def _create_bounding_boxes(self, y_pred, y):
        y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
        if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
            TP_total = 0
            FP_total = 0
            FN_total = 0
            N_total = 0
            N_pred_total = 0

            for m in range(y_pred.shape[0]):
                TP = 0

                labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
                labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)

                N_total += N
                N_pred_total += N_pred

                stats = cc3d.statistics(labels)
                stats_pred = cc3d.statistics(labels_pred)

                for i in range(1, N_pred + 1):
                    x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, \
                    stats_pred['bounding_boxes'][i][0].stop
                    y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, \
                    stats_pred['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N):
                        x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
                        y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP += 1
                            found = True
                        j += 1
                FP = N_pred - TP
                TP_rev = 0

                for i in range(1, N + 1):
                    x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
                    y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N_pred):
                        x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, \
                        stats_pred['bounding_boxes'][j][0].stop
                        y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, \
                        stats_pred['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP_rev += 1
                            found = True
                        j += 1

                FN = N - TP_rev

                TP_total += TP
                FP_total += FP
                FN_total += FN

            if (N_pred_total == 0) & (N_total != 0):
                return 0

            # Calculate complex metrics
            Recall = TP_total / (TP_total + FN_total)

        elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
            pass

        return Recall


class F1_all:
    def __init__(self,
                 batch_size=1,
                 spatial_dims=3,
                 bounding_box_dims=2,
                 spatial_dim_to_iter=0,
                 mode='mean'):

        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.bouding_box_dims = bounding_box_dims
        self.spatial_dim_to_iter = spatial_dim_to_iter
        self.mode = mode

    def __call__(self, y_pred, y):
        # Make sure dimensions match
        self._check_dimensions(y_pred=y_pred, y=y)

        # Argmax one-hot tensor
        y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)

        # Main loop for calculating the metric
        F1 = self.main_loop(y_pred=y_pred, y=y)

        return F1

    def main_loop(self, y_pred, y):
        F1 = []
        for b in range(self.batch_size):
            F1.append(self._create_bounding_boxes(y_pred=y_pred[b], y=y[b]))
        return F1

    def _check_dimensions(self, y_pred, y):
        if (len(y_pred.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
        elif (len(y.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')

    def _argmax_and_squeeze(self, y_pred, y):
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return y_pred, y

    def _create_bounding_boxes(self, y_pred, y):
        y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
        if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
            TP_total = 0
            FP_total = 0
            FN_total = 0
            N_total = 0
            N_pred_total = 0

            for m in range(y_pred.shape[0]):
                TP = 0

                labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
                labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)

                N_total += N
                N_pred_total += N_pred

                stats = cc3d.statistics(labels)
                stats_pred = cc3d.statistics(labels_pred)

                for i in range(1, N_pred + 1):
                    x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, \
                    stats_pred['bounding_boxes'][i][0].stop
                    y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, \
                    stats_pred['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N):
                        x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
                        y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP += 1
                            found = True
                        j += 1
                FP = N_pred - TP
                TP_rev = 0

                for i in range(1, N + 1):
                    x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
                    y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N_pred):
                        x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, \
                        stats_pred['bounding_boxes'][j][0].stop
                        y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, \
                        stats_pred['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP_rev += 1
                            found = True
                        j += 1

                FN = N - TP_rev

                TP_total += TP
                FP_total += FP
                FN_total += FN

            # Calculate complex metrics
            Precision = TP_total / (TP_total + FP_total)
            Recall = TP_total / (TP_total + FN_total)

            if (Precision == 0) & (Recall == 0):
                return 0

            F1 = 2 * ((Precision*Recall)/(Precision+Recall))

        elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
            pass

        return F1

class Precision:
    def __init__(self,
                 batch_size=1,
                 spatial_dims=3,
                 bounding_box_dims=2,
                 spatial_dim_to_iter=0,
                 mode='mean'):

        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.bouding_box_dims = bounding_box_dims
        self.spatial_dim_to_iter = spatial_dim_to_iter
        self.mode = mode

    def __call__(self, y_pred, y):
        # Make sure dimensions match
        self._check_dimensions(y_pred=y_pred, y=y)

        # Argmax one-hot tensor
        y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)

        # Main loop for calculating the metric
        Precision = self.main_loop(y_pred=y_pred, y=y)

        return Precision

    def main_loop(self, y_pred, y):
        Precision = []
        for b in range(self.batch_size):
            Precision.append(self._create_bounding_boxes(y_pred=y_pred[b], y=y[b]))
        return Precision

    def _check_dimensions(self, y_pred, y):
        if (len(y_pred.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
        elif (len(y.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')

    def _argmax_and_squeeze(self, y_pred, y):
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return y_pred, y

    def _create_bounding_boxes(self, y_pred, y):
        y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
        if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
            Precision = []

            for m in range(y_pred.shape[0]):
                TP = 0

                labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
                labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)

                # If no predictions were meade although there are lesions in GT
                if (N != 0) & (N_pred == 0):
                    Precision.append(0)
                    continue

                stats = cc3d.statistics(labels)
                stats_pred = cc3d.statistics(labels_pred)

                for i in range(1, N_pred + 1):
                    x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, \
                    stats_pred['bounding_boxes'][i][0].stop
                    y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, \
                    stats_pred['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N):
                        x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
                        y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP += 1
                            found = True
                        j += 1
                FP = N_pred - TP
                TP_rev = 0

                for i in range(1, N + 1):
                    x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
                    y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N_pred):
                        x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, \
                        stats_pred['bounding_boxes'][j][0].stop
                        y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, \
                        stats_pred['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP_rev += 1
                            found = True
                        j += 1

                FN = N - TP_rev

                # Calculate complex metrics
                Precision_m = TP / (TP + FP)

                Precision.append(Precision_m)
            match self.mode:
                case 'mean':
                    Precision = np.mean(Precision)
                case 'max':
                    Precision = np.max(Precision)
                case 'min':
                    Precision = np.min(Precision)

        elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
            pass

        return Precision


class Recall:
    def __init__(self,
                 batch_size=1,
                 spatial_dims=3,
                 bounding_box_dims=2,
                 spatial_dim_to_iter=0,
                 mode='mean'):

        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.bouding_box_dims = bounding_box_dims
        self.spatial_dim_to_iter = spatial_dim_to_iter
        self.mode = mode
    def __call__(self, y_pred, y):
        # Make sure dimensions match
        self._check_dimensions(y_pred=y_pred, y=y)

        # Argmax one-hot tensor
        y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)

        # Main loop for calculating the metric
        Recall = self.main_loop(y_pred=y_pred, y=y)

        return Recall

    def main_loop(self, y_pred, y):
        Recall = []
        for b in range(self.batch_size):
            Recall.append(self._create_bounding_boxes(y_pred=y_pred[b], y=y[b]))
        return Recall

    def _check_dimensions(self, y_pred, y):
        if (len(y_pred.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
        elif (len(y.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')

    def _argmax_and_squeeze(self, y_pred, y):
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return y_pred, y

    def _create_bounding_boxes(self, y_pred, y):
        y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
        if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
            Recall = []

            for m in range(y_pred.shape[0]):
                TP = 0

                labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
                labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)

                stats = cc3d.statistics(labels)
                stats_pred = cc3d.statistics(labels_pred)

                for i in range(1, N_pred + 1):
                    x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, stats_pred['bounding_boxes'][i][0].stop
                    y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, stats_pred['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N):
                        x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
                        y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP += 1
                            found = True
                        j += 1
                FP = N_pred - TP
                TP_rev = 0

                for i in range(1, N + 1):
                    x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
                    y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N_pred):
                        x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, stats_pred['bounding_boxes'][j][0].stop
                        y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, stats_pred['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP_rev += 1
                            found = True
                        j += 1

                FN = N - TP_rev

                # Calculate complex metrics
                Recall_m = TP / (TP + FN)

                Recall.append(Recall_m)
            match self.mode:
                case 'mean':
                    Recall = np.mean(Recall)
                case 'max':
                    Recall = np.max(Recall)
                case 'min':
                    Recall = np.min(Recall)

        elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
            pass

        return Recall


class F1:
    def __init__(self,
                 batch_size=1,
                 spatial_dims=3,
                 bounding_box_dims=2,
                 spatial_dim_to_iter=0,
                 mode='mean'
                 ):

        self.batch_size = batch_size
        self.spatial_dims = spatial_dims
        self.bouding_box_dims = bounding_box_dims
        self.spatial_dim_to_iter = spatial_dim_to_iter
        self.mode = mode

    def __call__(self, y_pred, y):
        # Make sure dimensions match
        self._check_dimensions(y_pred=y_pred, y=y)

        # Argmax one-hot tensor
        y_pred, y = self._argmax_and_squeeze(y_pred=y_pred, y=y)

        # Main loop for calculating the metric
        F1 = self.main_loop(y_pred=y_pred, y=y)

        return F1

    def main_loop(self, y_pred, y):
        F1 = []
        for b in range(self.batch_size):
            F1.append(self._create_bounding_boxes(y_pred=y_pred[b], y=y[b]))
        return F1

    def _check_dimensions(self, y_pred, y):
        if (len(y_pred.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y_pred is {len(y_pred.shape)}, expected {self.spatial_dims + 2}')
        elif (len(y.shape) != self.spatial_dims + 2):
            raise ValueError(f'Number of dimensions in y is {len(y.shape)}, expected {self.spatial_dims + 2}')

    def _argmax_and_squeeze(self, y_pred, y):
        y_pred = np.argmax(y_pred, axis=1)
        y = np.argmax(y, axis=1)
        return y_pred, y

    def _create_bounding_boxes(self, y_pred, y):
        y_pred, y = y_pred.astype('uint16'), y.astype('uint16')
        if self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 2D bounding boxes
            F1 = []

            for m in range(y_pred.shape[0]):
                TP = 0

                labels, N = cc3d.connected_components(y[m], connectivity=8, return_N=True)
                labels_pred, N_pred = cc3d.connected_components(y_pred[m], connectivity=8, return_N=True)

                stats = cc3d.statistics(labels)
                stats_pred = cc3d.statistics(labels_pred)

                for i in range(1, N_pred + 1):
                    x_start_pred, x_end_pred = stats_pred['bounding_boxes'][i][0].start, stats_pred['bounding_boxes'][i][0].stop
                    y_start_pred, y_end_pred = stats_pred['bounding_boxes'][i][1].start, stats_pred['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N):
                        x_start, x_end = stats['bounding_boxes'][j][0].start, stats['bounding_boxes'][j][0].stop
                        y_start, y_end = stats['bounding_boxes'][j][1].start, stats['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP += 1
                            found = True
                        j += 1
                FP = N_pred - TP
                TP_rev = 0

                for i in range(1, N + 1):
                    x_start, x_end = stats['bounding_boxes'][i][0].start, stats['bounding_boxes'][i][0].stop
                    y_start, y_end = stats['bounding_boxes'][i][1].start, stats['bounding_boxes'][i][1].stop
                    found = False
                    j = 1
                    while (not found) & (j <= N_pred):
                        x_start_pred, x_end_pred = stats_pred['bounding_boxes'][j][0].start, stats_pred['bounding_boxes'][j][0].stop
                        y_start_pred, y_end_pred = stats_pred['bounding_boxes'][j][1].start, stats_pred['bounding_boxes'][j][1].stop

                        if (x_start <= x_start_pred <= x_end) & (y_start <= y_start_pred <= y_end):
                            TP_rev += 1
                            found = True
                        j += 1

                FN = N - TP_rev

                # Calculate complex metrics
                # If no predictions were meade although there are lesions in GT
                if (N != 0) & (N_pred == 0):
                    Precision_m = 0
                else:
                    Precision_m = TP / (TP + FP)
                Recall_m = TP / (TP + FN)

                if (Precision_m == 0) & (Recall_m == 0):
                    F1.append(0)
                    continue

                F1_m = 2 * ((Precision_m * Recall_m) / (Precision_m + Recall_m))
                F1.append(F1_m)

            match self.mode:
                case 'mean':
                    F1 = np.mean(F1)
                case 'max':
                    F1 = np.max(F1)
                case 'min':
                    F1 = np.min(F1)

        elif self.spatial_dims - self.bouding_box_dims == 1:  # 3D data - 3D bounding boxes
            pass

        return F1
