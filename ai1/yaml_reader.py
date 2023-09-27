import yaml

def yaml_reader():
    with open("config.yaml","r") as file:
        config = yaml.safe_load(file)
    return config