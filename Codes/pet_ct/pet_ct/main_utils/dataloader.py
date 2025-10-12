"""
This file is a dataloader class which does the following:
- Creates a data list.
- Creates a dataset from the data list.
- Creates a dataloader to wrap the dataset.
"""

import monai
from pathlib2 import Path
from inspect import getfullargspec

class Dataloader:
    def __init__(self, args, transforms):
        self.args = args
        self.transforms = transforms

    def create_dataloader(self):
        for set in self.args.sets:
            self._create_data_list(set)
            self._create_dataset(set)
            self._create_dataloader(set)

    def _create_data_list(self, _T):
        json_path = Path(self.args.project_dir) / self.args.json_path
        data_list = monai.data.load_decathlon_datalist(json_path,
                                                       data_list_key=_T,
                                                       is_segmentation=getattr(self.args, 'is_segmentation', False),
                                                       base_dir=self.args.project_dir)
        setattr(self, _T + '_list', data_list)

    def _create_dataset(self, _T):
        # Get datalist
        dataList = getattr(self, _T + '_list')
        # Get Dataset Class
        dataset_conf_args = getattr(self.args, 'dataset_arguments')
        dataset_name = dataset_conf_args.get('dataset_type', 'Dataset')
        datasetFunc = getattr(monai.data, dataset_name)
        # Get arguments from config and check with the class's arguments
        dataset_conf_args.update({'data': dataList, 'transform':getattr(self.transforms, self.args.set2transforms[_T])})
        dataset_inspeced_args = getfullargspec(datasetFunc).args
        dataset_args = {k: dataset_conf_args[k] for k in dataset_conf_args if k in dataset_inspeced_args}
        # Initialize Dataset
        dataset = datasetFunc(**dataset_args)
        setattr(self, _T + '_dataset', dataset)

    def _create_dataloader(self, _T):
        shuffle_data = self.args.shuffle_data if _T == 'training' else False
        dataset = getattr(self, _T + '_dataset')
        dataloader = monai.data.DataLoader(dataset=dataset,
                                           batch_size=self.args.batch_size,
                                           shuffle=self.args.shuffle_data,
                                           collate_fn=monai.data.pad_list_data_collate)
        setattr(self, _T + '_dataloader', dataloader)
