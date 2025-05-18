import socket
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import sensor

row = 7
col = 5


def update_heatmap(new_data):
    heatmap.set_data(new_data)
    # 需要更新色彩条的最大最小值，如果数据变化幅度大的话
    
    fig.canvas.draw()
    fig.canvas.flush_events()
    
if __name__ == "__main__":

    # 初始化热力图
    plt.ion()  # 开启interactive mode
    fig, ax = plt.subplots()
    data0 = np.random.randint(309, 314, (row, col))
    heatmap = ax.imshow(data0, cmap='hot', interpolation='nearest')
    heatmap.set_clim(vmin=300, vmax=1000)
    # 添加颜色条
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label('Value')

    # 创建UDP socket
    local_ip = "127.0.0.1"
    local_port = 53000
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((local_ip, local_port))

    counter = 0

    while True:
        data, addr = sock.recvfrom(1024)  # 缓冲区大小为1024字节
        sensor_data = sensor.parse_sensor_data(data)
        pressure_sensors = sensor_data.pressure_sensors
        #print(pressure_sensors)
        if counter % 20 == 0:
            update_heatmap(np.reshape(pressure_sensors,(row, col)))
        
        counter += 1
