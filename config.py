influxdb = {
               "host_addr": "http://localhost:8086",
               "host_addr_ext": "http://x.x.x.x:8086",
               "token": "###token value###",
               "org": "watermonitor",
               "bucket": "compass"
           }
sensor = {
             "sampling_rate": 50,
             "pool_size": 500,
             "wait_between_two_pulls": 5,
             "accel_addr": 0x1c,
             "mag_addr": 0x6a
         }
file_paths = {
    "processed_data": ["../data/water_processed/", "../data/dataPool/08-07_processed/", "../data/dataPool/09-07_processed/", "../data/dataPool/10-07_processed/"],
    "data": ["../data/water/", "../data/dataPool/08-07/", "../data/dataPool/09-07/", "../data/dataPool/10-07/"],
    "csv_output_folder": "../data/dataPool/"
}
sine_fit = {
    "min_freq": 0.2,
    "max_freq": 20,
    "min_diff": 0.08,  # the min value diff between 2 points in an interval
    "min_amplitude": 0.03,
    "max_amplitude": 0.08
}