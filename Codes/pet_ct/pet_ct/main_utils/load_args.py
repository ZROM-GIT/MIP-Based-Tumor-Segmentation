"""
This file contains a function that load arguments given to it from a configuration file.
"""

import munch
import os 
from pathlib2 import Path
import yaml

from pet_ct.main_utils.my_utils import fix_args_if_needed



def load_args(conf, cli_args, project_dir):
    conf = getattr(cli_args, "config") or conf
    conf = Path(project_dir) / conf
    args = yaml.load(Path(conf).open('r'), Loader=yaml.CLoader)
    args = munch.Munch(args)
    args = fix_args_if_needed(args)
    args['project_dir'] = project_dir

    # Extract sub-project name
    subproject_name = (str(conf).split('/'))
    subproject_name = subproject_name[-2]
    args['subproject_name'] = subproject_name

    # Initialize experiment name
    Name = f'{args.project_name}{args.experiment_number}_{args.experiment_name}'
    args['full_experiment_name'] = Name
    return args
