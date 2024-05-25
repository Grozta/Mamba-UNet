import time
import random
from utils import ramps

# 定义一个装饰器函数
def calculate_function_run_time(func_name):
    """计算函数运行时间的装饰器"""
    def decorator_timer(func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            res = func(*args, **kwargs)
            end_time = time.time()
            run_time = end_time - start_time
            print("[{}]运行时间：".format(func_name), run_time, "秒")
            return res
        return wrapper
    return decorator_timer

@calculate_function_run_time("looper")
def looper(init_val):
    for e in range(init_val):
        print("{}".format(e))
        time.sleep(1)

def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "140": 1312}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]

def worker_init_fn(args,worker_id):
        random.seed(args.seed + worker_id)
        
def get_current_consistency_weight(args,consistency,epoch):  
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
        
def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)  

if __name__ == "__main__":
    looper(5)