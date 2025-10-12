"""
Create a list of masks to activate serially
"""

from .HGUO_mask import HGUO_mask


class masks:
    def __init__(self, args):
        self.args = args
        self.masks_args = self.args['masks']

        self.masks_name_to_class = {
            'HGUO_mask': HGUO_mask
        }

        self._create_instances_of_masks()

    def _create_instances_of_masks(self):
        self.list_of_masks = []
        for mask_name in self.masks_args:
            mask_args = self.masks_args[f'{mask_name}']
            mask_class = self.masks_name_to_class[f'{mask_name}']
            mask = mask_class(mask_args)
            self.list_of_masks.append(mask)

    def __call__(self, input, output):
        for mask_fun in self.list_of_masks:
            output = mask_fun(input=input, output=output)
        return output


