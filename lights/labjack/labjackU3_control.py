import os, sys, serial, time
import u3

def blink_led(d):

    d.toggleLED()
    time.sleep(0.1)
    d.toggleLED()

def set_DAC(d,DAC_num,voltage):
    blink_led(d)

    print('Setting DAC' + str(DAC_num) + ' to ' + str(voltage))

    dac_value = int((voltage / 5.0) * 65535)  # Convert voltage to DAC value

    if DAC_num == 0:
        d.writeRegister(5000, dac_value)  # Write DAC0 value to register 5000
    elif DAC_num == 1:
        d.writeRegister(5002, dac_value)  # Write DAC1 value to register 5002


if __name__ == "__main__":
    d = u3.U3()
    print(d.configU3())

    set_DAC(d,0,0)
    set_DAC(d,1,0)

    set_DAC(d,0,5)
    time.sleep(5)
    set_DAC(d,0,0)

    set_DAC(d,1,5)
    time.sleep(5)
    set_DAC(d,1,0)

    print(round(d.getTemperature()-273.15,1))
    blink_led(d)

    d.close()