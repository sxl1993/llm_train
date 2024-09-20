import os
import sys
from transformers import HfArgumentParser
from transformers.training_args import TrainingArguments
from typing import Optional, Dict, Any

from arguments import FeatureExtractionArguments, DatasetArguments, ModelArguments

_TRAIN_CLS = [FeatureExtractionArguments, DatasetArguments, ModelArguments, TrainingArguments]


# 将 YAML 配置文件解析为对应的参数类
def _parse_args(parser: "HfArgumentParser", args: Optional[Dict[str, Any]] = None):
    if args is not None:
        return parser.parse_dict(args)
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
        return parser.parse_yaml_file(os.path.abspath(sys.argv[1]))
    
    (*parsed_args, unknown_args) = parser.parse_args_into_dataclasses(return_remaining_strings=True)
    
    if unknown_args:
        print(parser.format_help())
        print("Got unknown args, potentially deprecated arguments: {}".format(unknown_args))
        raise ValueError("Some specified arguments are not used by the HfArgumentParser: {}".format(unknown_args))
      
    return (*parsed_args,)
    
def parse_train_args(args: Optional[Dict[str, Any]] = None) -> _TRAIN_CLS:
    parser = HfArgumentParser(_TRAIN_CLS)
    return _parse_args(parser, args)