import os
def select_gpus(number:str):
    #示例：os.environ ["CUDA_VISIBLE_DEVICES"] = "0，1"；os.environ ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = number