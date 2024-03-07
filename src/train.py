from configs.config import params
import pprint
import yaml 

# pprint.pprint(params)

with open("./configs/yolo-m-mish.yaml", 'r') as f:
    yolo_m = yaml.load(f, Loader=yaml.FullLoader)
pprint.pprint(yolo_m)