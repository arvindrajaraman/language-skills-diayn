import pathlib

import ruamel.yaml as yaml

root = pathlib.Path(__file__).parent
y = yaml.YAML(typ='safe')
for key, value in y.load((root / 'data_text.yaml').read_text()).items():
  globals()[key] = value
