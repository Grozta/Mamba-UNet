import time

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


if __name__ == "__main__":
    looper(5)