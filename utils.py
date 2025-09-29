

class CustomEvaluator(object):
    def __init__(self, obj_fun):
        self.iteration_count = 0  # 반복 횟수 카운터 초기화
        self.obj_fun = obj_fun

    def evaluate(self, x):
        print(f"현재 BO Iteration: {self.iteration_count}")
        val = self.obj_fun(x)
        self.iteration_count += 1
        return val
