import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

# Assume the following parameter definitions exist in param.py
# Modify them based on actual requirements
class param:
    data_dir = './exp/1212/26r'          # Directory containing raw data files (e.g., 500g-1.csv, 500g-2.csv, etc.)
    output_csv = f'{data_dir}/output.csv'  # Final output file for regression results

############################
# Functions from cali_act.py #
############################

def sma_filter(input, window_size=10):
    result = []

    if len(input) < window_size:
        raise ValueError("SMA filter error: Data length is less than the window size.")

    # Start phase: Window grows from 1 to window_size
    for i in range(0, window_size):
        window = input[:i + 1]
        sma = sum(window) / (i + 1)
        result.append(sma)

    # Middle phase: Window size is fixed to window_size
    for i in range(window_size, len(input) - window_size):
        window = input[i:i + window_size]
        sma = sum(window) / window_size
        result.append(sma)

    # End phase: Window shrinks from window_size down to 1
    for i in range(len(input) - window_size, len(input)):
        window = input[i:]
        sma = sum(window) / (len(input) - i)
        result.append(sma)

    return result

def read_pressure_data_from_csv(filepath, p_num=25):
    pressure_data_list = []

    with open(filepath, 'r', newline='') as csvfile:
        reader = csv.DictReader(csvfile)

        # 获取列名并检查是否有 Timestamp 列
        fieldnames = reader.fieldnames
        if fieldnames[0] != 'Timestamp':
            # 如果第一列不是 Timestamp，则跳过一列重新读取
            #next(reader)
            reader = csv.DictReader(csvfile)
        for row in reader:
            # Extract timestamp and its millisecond part
            timestamp = float(row['Timestamp'])
            # Extract sensor values
            pressure_sensors = [int(row[f'P{i}']) for i in range(1, p_num + 1)]

            # Add it to the list
            pressure_data_list.append(pressure_sensors)

    return pressure_data_list

def draw_pressure(pressure_data, title='Pressure Value', xlabel='Package Count', ylabel='Voltage Value'):
    """
    Visualize pressure sensor data.
    """
    for i in range(len(pressure_data)):
        plt.plot(pressure_data[i], label=f'P{i+1}')
    plt.legend()
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

def find_activation_mean(data, percent_threshold=0.75, min_duration=150):
    """
    Find the activation mean for each sensor based on the threshold and minimum duration.
    """
    num_sensors, num_samples = data.shape
    activation_means = []
    activation_durations = []
    activation_diffs = []

    for sensor in range(num_sensors):
        sensor_data = data[sensor, :]
        min_value = np.average(sensor_data[:500])
        max_value = np.max(sensor_data)
        threshold = (max_value - min_value) * percent_threshold + min_value

        activation_start = -1
        activation_end = -1

        # Identify the start and end of the activation period
        for i in range(num_samples):
            if sensor_data[i] > threshold:
                if activation_start == -1:
                    activation_start = i
            else:
                if activation_start != -1:
                    if i - activation_start >= min_duration:
                        activation_end = i
                        break
                    else:
                        activation_start = -1

        if activation_start != -1 and activation_end != -1:
            activation_mean = np.mean(sensor_data[activation_start:activation_end])
            activation_duration = activation_end - activation_start
            activation_diff = max_value - activation_mean 
        else:
            activation_mean = np.nan
            activation_duration = 0
            activation_diff = 0

        activation_means.append(activation_mean)
        activation_durations.append(activation_duration)
        activation_diffs.append(activation_diff)

    return np.array(activation_means), np.array(activation_durations), np.array(activation_diffs)

def calculate_activation_values(pressure_list):
    """
    Calculate activation values for sensors using SMA filtering and thresholding.
    """
    pressure_sma = []
    for i in pressure_list:
        pressure_sma.append(sma_filter(i, 100))

    activation_values = find_activation_mean(np.array(pressure_sma),0.75,50)
    activation_real, activation_durations, activation_diffs = activation_values

    #draw_pressure(pressure_sma, title="SMA Pressure")
    return activation_real, activation_durations

def v_list_to_r(v_list):
    """
    Convert voltage values (V) to resistance values (R) using reference parameters.
    """
    v_ref = 0.312
    R1 = 5000
    for i in range(len(v_list)):
        if v_list[i] > v_ref:
            v_list[i] = R1 * v_ref / (v_list[i] - v_ref)
        else:
            v_list[i] = float('inf')
    return v_list

##############################
# Functions from cali_recur.py #
##############################

def r_f_recur_calculate(data_df, output_csv):
    """
    Perform regression to calculate parameters for R = k * F^alpha.
    """
    R_values_507g = data_df.loc['507g(R)'].values.astype(float)
    R_values_1009g = data_df.loc['1009g(R)'].values.astype(float)
    F_values_507g = data_df.loc['507g(F)'].values.astype(float)
    F_values_1009g = data_df.loc['1009g(F)'].values.astype(float)

    results = []
    for sensor_id in range(len(R_values_507g)):
        R = np.array([R_values_507g[sensor_id], R_values_1009g[sensor_id]])
        F = np.array([F_values_507g[sensor_id], F_values_1009g[sensor_id]])

        if len(R) == 2 and len(F) == 2:
            log_R = np.log(R)
            log_F = np.log(F)

            A = np.vstack([log_F, np.ones(len(log_F))]).T
            alpha, log_k = np.linalg.lstsq(A, log_R, rcond=None)[0]
            k = np.exp(log_k)
            results.append({'Sensor': sensor_id+1, 'k': k, 'alpha': alpha})

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, index=False)

########################################
# Main process for automation          #
########################################

def process_files_for_load(directory, load_pattern, sensor_num=35):
    """
    Process multiple files for the same load (e.g., 500g or 1kg).
    Calculate the activation resistance (R) and return the average R values for the load.
    """
    pattern = re.compile(fr"{load_pattern}-\d+\.csv$")
    R_list_per_file = []

    for filename in os.listdir(directory):
        if pattern.search(filename):
            input_csv = os.path.join(directory, filename)
            pressure_time_list = np.array(read_pressure_data_from_csv(input_csv, sensor_num))
            pressure_list = pressure_time_list.T
            activation_real, _ = calculate_activation_values(pressure_list)
            activation_resistance = v_list_to_r(activation_real / 1000.0)
            R_list_per_file.append(activation_resistance)

    if len(R_list_per_file) == 0:
        raise ValueError(f"No files matching {load_pattern} were found.")

    R_list_per_file = np.array(R_list_per_file)
    avg_R = np.nanmean(R_list_per_file, axis=0)
    return avg_R


def main():
    """
    Main entry point for processing sensor data and performing regression.
    """
    sensor_num = 35

    # Patterns for file naming based on load
    load1_pattern = "500g"
    load2_pattern = "1kg"

    R_500g = process_files_for_load(param.data_dir, load1_pattern, sensor_num=sensor_num)
    R_1kg  = process_files_for_load(param.data_dir, load2_pattern, sensor_num=sensor_num)

    F_500g = 4.967
    F_1kg = 9.888

    sensors = list(range(sensor_num))
    data = {
        '507g(R)': R_500g,
        '1009g(R)': R_1kg,
        '507g(F)': np.array([F_500g]*sensor_num),
        '1009g(F)': np.array([F_1kg]*sensor_num)
    }

    df_for_recur = pd.DataFrame(data, index=sensors).T
    r_f_recur_calculate(df_for_recur, param.output_csv)

    print("Regression completed. Results saved to:", param.output_csv)


if __name__ == "__main__":
    main()
