import csv
import os
import sys

import config as cfg
from ProducerConsumer import ConsumerThread, ProducerThread

if os.uname().nodename == 'raspberrypi':
    # python3 -m pip install --upgrade setuptools
    # python3 -m pip install --upgrade pip
    # python3 -m pip install influxdb-client
    import board  # python3 -m pip install adafruit-blinka --no-cache-dir

# sudo apt-get install python3-pandas
import time
import busio
import threading
import numpy as np
from LSM9DS1_I2C_Sensor import Lsm9sd1I2CSensor
from influxdb_client import InfluxDBClient
from influxdb_client.client.write_api import SYNCHRONOUS

"""
This module provides different methods to retrieve the data from the Lsm9sd1I2C connected on the I2C bus:
    - print_data() to print on the standard output data from the sensor
    - save_data() to save data from the sensor onto the database
    - get_data_as_csv() to save as a csv file data from the database

"""


def _connect_to_i2c():
    sampling_rate = cfg.sensor["sampling_rate"]
    accel_addr = cfg.sensor["accel_addr"]
    mag_addr = cfg.sensor["mag_addr"]
    if sampling_rate > 1000:
        raise ValueError("Sampling rate of magnetometer should not be greater than 1000Hz")
    if accel_addr not in (0x1c, 0x1e):
        raise ValueError("Accelerometer should be either on address 0x1c or 0x1e")
    if mag_addr not in (0x6a, 0x6b):
        raise ValueError("Magnetometer should be either on address 0x6a or 0x6b")

    i2c = busio.I2C(board.SCL, board.SDA)
    sensor = Lsm9sd1I2CSensor(i2c, accel_addr, mag_addr)
    sensor.set_mag_to_fast_odr()

    return sensor


def _connect_to_influxdb(host_addr):
    client = InfluxDBClient(url=host_addr, token=cfg.influxdb["token"])
    return client


def _to_ns(timestamp):
    # a timestamp in ns has a length of 19
    len_diff = 19 - len(str(timestamp))
    if len_diff >= 0:
        return timestamp * 10 ** len_diff
    else:
        raise ValueError("Timestamp value is too big or uses incorrect unit (should not be more precise than ns).")


def get_data_as_csv(from_timestamp, to_timestamp, host_addr=cfg.influxdb["host_addr_ext"], interval=None):
    """
    Query the data from the database to save it as csv file
    :param from_timestamp: start of the interval the data should be extracted
    :param to_timestamp: end of the interval the data should be extracted
    :param host_addr: address the database
    :param filename: name of the csv file that will be created, by default it will be named from_timestamp-to_timestamp.csv
    :param interval: maximum time (in seconds) recorded in the file
    """
    from_timestamp = _to_ns(from_timestamp)
    to_timestamp = _to_ns(to_timestamp)
    if interval is None:
        interval = to_timestamp - from_timestamp
    else:
        interval = interval * 10 ** 9

    client = _connect_to_influxdb(host_addr)
    sub_to_timestamp = 0
    while sub_to_timestamp < to_timestamp:
        sub_to_timestamp = min(to_timestamp, from_timestamp + interval)
        print(str(from_timestamp) + " to " + str(sub_to_timestamp))
        filename = cfg.file_paths["csv_output_folder"] + str(from_timestamp) + "-" + str(sub_to_timestamp) + '.csv'

        query = 'from(bucket:"' + cfg.influxdb["bucket"] + '")'
        query += '|> range(start:time(v:' + str(from_timestamp) + '),stop:time(v:' + str(sub_to_timestamp) + '))'
        query += '|> filter(fn:(r) => r._field == "norm")'
        # query += '|> map(fn:(r) => ({ r with exposures: uint(v: r.exposures) }))'
        result = client.query_api().query(org=cfg.influxdb["org"], query=query)

        with open(filename, 'w') as output:
            writer = csv.writer(output, delimiter=',')
            writer.writerow(['time', 'value'])
            for table in result:
                for record in table.records:
                    writer.writerow([record.get_time(), record.get_value()])

        from_timestamp = sub_to_timestamp


def save_data():
    """
    Starts the recording of the data retrieved from the sensor into the database
    """
    sensor = _connect_to_i2c()
    client = _connect_to_influxdb(cfg.influxdb["host_addr"])
    waiting_time = 1 / cfg.sensor["sampling_rate"]

    # warmup
    print("### WARMUP")
    for i in range(0, 1000):
        sensor.magnetic
    print("### RECORDING DATA")

    def _prod_fct():
        mag_x, mag_y, mag_z = sensor.magnetic
        t = time.time_ns()
        line = str(cfg.influxdb["bucket"]) + ' norm=' + str(np.linalg.norm([mag_x, mag_y, mag_z]))
        linexyz = ',x=' + str(mag_x) + ',y=' + str(mag_y) + ',z=' + str(mag_z)
        # default precision of influxdb2.0 is ns
        timestamp = " " + str(t)
        return line + linexyz + timestamp

    def _consum_fct(q, pool_size):
        data = []
        for i in range(0, pool_size):
            data.append(q.get())
        write_api = client.write_api(write_options=SYNCHRONOUS)
        try:
            write_api.write(cfg.influxdb["bucket"], cfg.influxdb["org"], data)
        except Exception as e:
            print(e)
            sys.exit(1)

    p = ProducerThread(_prod_fct, name='producer', time_between_samples=waiting_time)
    c = ConsumerThread(_consum_fct, name='consumer', pool_size=cfg.sensor["pool_size"],
                       wait_between_two_pulls=cfg.sensor["wait_between_two_pulls"])
    p.start()
    c.start()


def print_data():
    """
    Starts the printing of the data retrieved from the sensor
    """
    sensor = _connect_to_i2c()
    lag_time_print = 0.0005  # time lag when printing
    waiting_time = 1/cfg.sensor["sampling_rate"] - lag_time_print

    # warmup
    print("### WARMUP")
    for i in range(0, 1000):
        sensor.magnetic
    print("### RECORDING DATA")
    while True:
        t = threading.Thread(target=time.sleep, args=(waiting_time,), daemon=True)
        t.start()
        # START THREAD ####
        mag_x, mag_y, mag_z = sensor.magnetic
        timestamp = time.time() * 10**3
        print(str(timestamp) + " " + str(np.linalg.norm([mag_x, mag_y, mag_z])))
        t.join()
        # END THREAD ######
