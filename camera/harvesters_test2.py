from harvesters.core import Harvester
import matplotlib.pyplot as plt
import cv2

h = Harvester()
h.add_file(r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti")

h.update()

h.device_info_list

ia = h.create(0)
# ia = h.create({'serial_number': 'SN_InterfaceA_0'})
print(h.device_info_list[0].search_keys)





# start acquiring images
ia.start()

for i in range(1000):
    with ia.fetch() as buffer:
        # Work with the Buffer object. It consists of everything you need.
        print(buffer,i)
        # The buffer will automatically be queued.

ia.stop()