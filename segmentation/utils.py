import yaml
from yaml.loader import SafeLoader
from addict import Dict as ADict


def config_parser(config_path) -> ADict:
    # find .yaml file by experiment name
    with open(config_path, "r", encoding="utf-8") as yaml_file:
        config_data = yaml.load(yaml_file, Loader=SafeLoader)
    return ADict(config_data)
