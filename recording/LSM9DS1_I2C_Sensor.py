import adafruit_lsm9ds1


class Lsm9sd1I2CSensor(adafruit_lsm9ds1.LSM9DS1_I2C):
    def set_mag_to_fast_odr(self):
        self.set_property_mag(adafruit_lsm9ds1._LSM9DS1_REGISTER_CTRL_REG1_M, 0b10)

    def set_property_mag(self, address, val):
        self._write_u8(adafruit_lsm9ds1._MAGTYPE, address, val)

    def set_property_accel(self, address, val):
        self._write_u8(adafruit_lsm9ds1._XGTYPE, address, val)
