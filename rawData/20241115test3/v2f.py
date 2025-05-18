import os
import pandas as pd
from sensor import read_sensor_data_from_csv, save_sensor_data_to_csv

# 加载传感器的拟合参数
def load_fitting_parameters(true_recur):
    df = pd.read_csv(true_recur)
    params = {row['Sensor']: (row['k'], row['alpha']) for _, row in df.iterrows()}
    return params

# 主函数：批量处理目录中的CSV文件
def batch_process_csv_files():
    exp = 'InsoleSensor'
    left_recur_results_csv = f"./{exp}/left.csv"
    right_recur_results_csv = f"./{exp}/right.csv"
    output_dir = "output"
    os.makedirs(output_dir, exist_ok=True)

    # 遍历当前目录和子目录中的CSV文件
    for root, _, files in os.walk(f"./{exp}/"):
        for file in files:
            if file.endswith(".csv") and file not in ["left.csv", "right.csv"]:
                if "left" in file:
                    recur_results_csv = left_recur_results_csv
                elif "right" in file:
                    recur_results_csv = right_recur_results_csv
                else:
                    continue
                
                # 加载拟合参数
                params = load_fitting_parameters(recur_results_csv)
                
                # 文件路径处理
                sensor_input_path = os.path.join(root, file)
                relative_path = os.path.relpath(sensor_input_path, ".")
                output_path = os.path.join(output_dir, relative_path)
                os.makedirs(os.path.dirname(output_path), exist_ok=True)

                # 读取传感器数据
                sensor_data_list = read_sensor_data_from_csv(sensor_input_path)
                
                # # 使用sensor模块的v_to_r和r_to_f方法进行转换
                # for sensor_data in sensor_data_list:
                #     sensor_data.sensor_v_to_r()  # 电压转电阻
                #     sensor_data.sensor_r_to_f(params)  # 电阻转压力

                for sensor_data in sensor_data_list:
                    print(f"Original Voltage Data: {sensor_data.pressure_sensors[1]}")
                    sensor_data.sensor_v_to_r()  # 電圧から電阻への変換
                    print(f"After V to R: {sensor_data.pressure_sensors[1]}")
                    sensor_data.sensor_r_to_f(params)  # 電阻から圧力への変換
                    print(f"After R to F: {sensor_data.pressure_sensors[1]}")



                # 保存转换后的数据
                save_sensor_data_to_csv(sensor_data_list, output_path)
                print(f"Processed {file} and saved to {output_path}")

# 执行批量处理
if __name__ == "__main__":
    batch_process_csv_files()
