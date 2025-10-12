"""
This file contains a model class that finds the model class noted in conf. file from a list of models.
It checks which arguments the model accepts and checks mutual arguments from configuration file.
"""

import inspect
import monai.networks.nets as nets

from pet_ct.models.model_name_to_class import models
from pet_ct.models.my_models.ViT_pytorch.models.configs import get_mip2d_config
from pet_ct.models.my_models.ViT_pytorch.models.modeling import VisionTransformer

class Model:
    def __init__(self, args):
        self.args = args
        self.model_args = getattr(self.args, 'model_arguments')
        self.model_name = self.model_args['model_name']

    def create_model(self):
        if self.model_name in models:  # Check if model name is in my models
            self.model = models[self.model_name]
        elif hasattr(nets, self.model_name):  # Check if model name appears in monai networks
            self.model = getattr(nets, self.model_name)
        else:
            self.model = VisionTransformer(config=get_mip2d_config(), img_size=(704, 400))
            return self.model
        self._get_input_args_from_model()
        self._get_mutual_args()
        return self.model(**self.modelArgs)

    def _get_input_args_from_model(self):
        inputArgs = list(inspect.signature(self.model).parameters)
        setattr(self, 'inputArgs', inputArgs)

    def _get_mutual_args(self):
        modelArgs = {k: self.model_args[k] for k in self.model_args if k in self.inputArgs}
        setattr(self, 'modelArgs', modelArgs)



