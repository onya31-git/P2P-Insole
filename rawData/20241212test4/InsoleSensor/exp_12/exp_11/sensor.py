import struct
import csv
import numpy as np

coordinate_x_35_insole = [
    -40.6, -21.2, -6.5, 7.2, 17.3,
    -39.6, -24.3, -8.2, 4.3, 15.2,
    -35.3, -23, -8.9, 4.5, 16.2,
    -17, -8.5, 0, 8.5, 17,
    -19.5, -11, -2.5, 6, 14.5,
    -29, -19, -9, 1, 11,
    -30, -20.5, -11, -1.5, 8
]

coordinate_y_35_insole = [
    -100.5, -104, -100.7, -88, -73,
    -70.8, -68.9, -65, -59.8, -54,
    -39, -36, -32, -28.5, -25.2,
    0, 0, 0, 0, 0,
    40, 40, 40, 40, 40,
    70, 70, 70, 70, 70,
    90, 90, 90, 90, 90,
]

# 传感器读数（每帧）
class SensorData:
    def __init__(self, timestamp, pressure_sensors, magnetometer, gyroscope, accelerometer):
        self.timestamp = timestamp
        self.pressure_sensors = pressure_sensors  # 这里仍然用pressure_sensors来表示电阻值
        self.magnetometer = magnetometer
        self.gyroscope = gyroscope
        self.accelerometer = accelerometer
    
    def sensor_v_to_r(self):
        v_ref = 0.312
        R1 = 5000
        for i in range(len(self.pressure_sensors)):
            current_v = self.pressure_sensors[i] / 1000
            if current_v > v_ref:
                self.pressure_sensors[i] = R1 * v_ref / (current_v - v_ref)
            else:
                self.pressure_sensors[i] = float('inf')

    def sensor_r_to_f(self, params):
        for i in range(len(self.pressure_sensors)):
            sensor_id = i + 1
            if sensor_id in params:
                k, alpha = params[sensor_id]
                R = self.pressure_sensors[i]
                if R != float('inf'):
                    res = (R / k) ** (1 / alpha)
                    if res < 1e-2:
                        self.pressure_sensors[i] = 0  # 处理压力值过小的情况
                    elif res > 50:
                        self.pressure_sensors[i] = 50 # 处理异常值或超过量程的情况
                    else:
                        self.pressure_sensors[i] = res
                else:
                    self.pressure_sensors[i] = 0  # 处理电阻无限大的情况


# 传感器数据
class SensorDataList:
    def __init__(self, sensor_data_list):
        self.sensor_data_list = sensor_data_list
    
    # 提取加速度函数
    def get_acc(self):
        acc_x = [data.accelerometer[0] for data in self.sensor_data_list]
        acc_y = [data.accelerometer[1] for data in self.sensor_data_list]
        acc_z = [data.accelerometer[2] for data in self.sensor_data_list]
        return [acc_x, acc_y, acc_z]
    
    # 提取角速度函数
    def get_gyro(self):
        gyro_x = [data.gyroscope[0] for data in self.sensor_data_list]
        gyro_y = [data.gyroscope[1] for data in self.sensor_data_list]
        gyro_z = [data.gyroscope[2] for data in self.sensor_data_list]
        return [gyro_x, gyro_y, gyro_z]

    # 提取磁力计函数
    def get_mag(self):
        mag_x = [data.magnetometer[0] for data in self.sensor_data_list]
        mag_y = [data.magnetometer[1] for data in self.sensor_data_list]
        mag_z = [data.magnetometer[2] for data in self.sensor_data_list]
        return [mag_x, mag_y, mag_z]

    # 提取时间戳函数
    def get_timestamp(self):
        timestamp = [data.timestamp for data in self.sensor_data_list]
        return timestamp

    # 提取压力矩阵函数
    def get_pressure(self):
        pressure = [data.pressure_sensors for data in self.sensor_data_list]
        return pressure
    
    # 提取压力和函数
    def get_pressure_sum(self):
        pressure_sum = []
        for i in self.sensor_data_list:
            pressure_sum.append(np.sum(i.pressure_sensors))
        return pressure_sum

    # 提取COP函数
    def get_pressure_cop(self):
        pressure_x_cop = []
        pressure_y_cop = []
        for i in self.sensor_data_list:
            for j in range(len(i.pressure_sensors)):
                weight_coordinate_x = []
                weight_coordinate_x.append(coordinate_x_35_insole[j] * i.pressure_sensors[j])
                weight_coordinate_y = []
                weight_coordinate_y.append(coordinate_y_35_insole[j] * i.pressure_sensors[j])

            pressure_y_cop.append(np.sum(weight_coordinate_y) / np.sum(i.pressure_sensors))
            pressure_x_cop.append(np.sum(weight_coordinate_x) / np.sum(i.pressure_sensors))

        return pressure_x_cop, pressure_y_cop
                


# 收集数据函数（逐个）
# ESP32的二进制数据 -> SensorData对象
# 解析传入的二进制数据，返回SensorData对象
def parse_sensor_data(data):
    # 首先检查起始和结束标志是否正确
    if data[:2] == b'\x5a\x5a' and data[-2:] == b'\xa5\xa5':
        # 解析DN和SN字段
        dn, sn = struct.unpack('BB', data[2:4])
        # 解析时间戳（4字节整数）
        timestamp = struct.unpack('<I', data[4:8])[0]
        # 解析时间戳毫秒（2字节整数）
        timems = struct.unpack('<H', data[8:10])[0]
        # 计算除压力传感器外的固定长度部分
        # 起始2 DN1 SN1 时间戳4+2 三轴传感器3*12 终止2
        fixed_length = 2 + 1 + 1 + 4 + 2 + 12 + 12 + 12 + 2
        # 计算压力传感器数据长度，从而确定压力传感器数量
        number_of_pressure_sensors = (len(data) - fixed_length) // 4
        # 设置压力传感器数据的起始位置
        pressure_start_position = 10
        # 解析每个压力传感器的整数值
        pressure_sensors = [struct.unpack('<i', data[pressure_start_position + i * 4:pressure_start_position + (i + 1) * 4])[0]
                            for i in range(number_of_pressure_sensors)]
        # 设置和解析磁力计、陀螺仪、加速度计数据的起始和结束位置
        magnetometer_start = pressure_start_position + number_of_pressure_sensors * 4
        magnetometer = struct.unpack('<3f', data[magnetometer_start:magnetometer_start + 12])
        gyroscope_start = magnetometer_start + 12
        gyroscope = struct.unpack('<3f', data[gyroscope_start:gyroscope_start + 12])
        accelerometer_start = gyroscope_start + 12
        accelerometer = struct.unpack('<3f', data[accelerometer_start:accelerometer_start + 12])
        
        return SensorData(timestamp + timems/1000, pressure_sensors, magnetometer, gyroscope, accelerometer)
    # 忽略标志错误的数据包
    else:
        return None

# 收集数据函数
# SensorData对象列表 -> CSV
def save_sensor_data_to_csv(sensor_data_list, filename):
    # 保存传感器数据到CSV文件
    with open(filename, 'w', newline='') as csvfile:
        # 创建CSV文件写入器
        header_writer = csv.writer(csvfile)

        # 首先检查数据列表是否为空，以避免索引错误
        if len(sensor_data_list) != 0:
            # 写入DN和SN
            header_writer.writerow([f"// DN: {108}, SN: {35}"])

        # 定义CSV文件的列标题
        fieldnames = ['Timestamp'] + [f'P{i + 1}' for i in range(len(sensor_data_list[0].pressure_sensors))] + \
                     ['Mag_x', 'Mag_y', 'Mag_z', 'Gyro_x', 'Gyro_y', 'Gyro_z', 'Acc_x', 'Acc_y', 'Acc_z']
        
        # 创建字典写入器，并写入列标题
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 遍历传感器数据列表， 写入数据到CSV文件
        for sensor_data in sensor_data_list:
            row_data = {'Timestamp': sensor_data.timestamp}
            row_data.update({f'P{i + 1}': sensor_data.pressure_sensors[i] for i in range(len(sensor_data.pressure_sensors))})
            row_data.update({'Mag_x': sensor_data.magnetometer[0], 'Mag_y': sensor_data.magnetometer[1], 'Mag_z': sensor_data.magnetometer[2],
                             'Gyro_x': sensor_data.gyroscope[0], 'Gyro_y': sensor_data.gyroscope[1], 'Gyro_z': sensor_data.gyroscope[2],
                             'Acc_x': sensor_data.accelerometer[0], 'Acc_y': sensor_data.accelerometer[1], 'Acc_z': sensor_data.accelerometer[2]})
            writer.writerow(row_data)

# 读取数据函数
# CSV -> 数据列表
def read_sensor_data_from_csv(filepath, p_num=35):
    sensor_data_list = []

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
            magnetometer = [float(row[f'Mag_{axis}']) for axis in ['x', 'y', 'z']]
            gyroscope = [float(row[f'Gyro_{axis}']) for axis in ['x', 'y', 'z']]
            accelerometer = [float(row[f'Acc_{axis}']) for axis in ['x', 'y', 'z']]
            
            # Create a SensorData instance and add it to the list
            sensor_data = SensorData(timestamp, pressure_sensors, magnetometer, gyroscope, accelerometer)
            sensor_data_list.append(sensor_data)

    return sensor_data_list

# 测试函数
def test_save():
    data_example = b'ZZ\xe0\n\xd6w8f\xb7\x017\x01\x00\x008\x01\x00\x008\x01\x00\x008\x01\x00\x007\x01\x00\x009\x01\x00\x00:\x01\x00\x00:\x01\x00\x00:\x01\x00\x00;\x01\x00\x00\x00\x00\x80@\x00\x00`A\x00\x00\x1cB\xff\xffy=\xff\xff\xf9\xbd\x00\x00\x00\x00\x00\x00U=\x00\x00\xd3\xbc\x00\xe0~?\xa5\xa5'
    sensor_data = parse_sensor_data(data_example)

    if sensor_data:
        # 将数据保存到CSV文件
        save_sensor_data_to_csv([sensor_data], 'sensor_data.csv')

def test_read():
    return read_sensor_data_from_csv("./sensor_data.csv", 10)

if __name__ == "__main__":
    # 示例数据
    test_save()
    a = test_read()
    print(a)