import pandas as pd
import numpy as np
import os

class CustomEvaluator(object):
    def __init__(self, obj_fun, excel_path, var_names):
        self.iteration_count = 0  # 반복 횟수 카운터 초기화
        self.obj_fun = obj_fun

        self.excel_path = excel_path
        self.row_names = ["Iter"] + var_names
        self.data = pd.DataFrame(columns=self.row_names)

    def evaluate(self, x):
        print(f"현재 BO Iteration: {self.iteration_count}")
        val, optional_res = self.obj_fun(x)

        data_row = np.concatenate(([[self.iteration_count]], x, [[val]], optional_res), axis=1)
        self.data = self.data.append(pd.DataFrame(data_row, columns=self.row_names), ignore_index=True)

        writer_options = self._get_writer_options()
        with pd.ExcelWriter(self.excel_path, **writer_options) as writer:
            self.data.to_excel(writer, index=False, sheet_name="BO_Result")

        self.iteration_count += 1
        return val

    def save_best_result(self):
        best_row = self.data.nsmallest(1, "OBJ")
        writer_options = self._get_writer_options()
        with pd.ExcelWriter(self.excel_path, **writer_options) as writer:
            best_row.to_excel(writer, index=False, sheet_name="Best_Result")

    def read_past_results(self, past_path):
        try:
            with pd.ExcelFile(past_path) as xls:
                df = pd.read_excel(xls, sheet_name="BO_Result")
            return df
        except Exception:
            return pd.DataFrame(columns=self.row_names)

    def _get_writer_options(self):
        if os.path.exists(self.excel_path):
            writer_options = dict(mode='a', engine='openpyxl', if_sheet_exists='replace')
        else:
            writer_options = {}
        return writer_options