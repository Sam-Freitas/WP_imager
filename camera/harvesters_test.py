import cv2
from harvesters.core import Harvester

def imshow_resize(frame_name = "img", frame = 0, resize_size = [640,480], default_ratio = 1.3333, always_on_top = True, use_waitkey = True):

    frame = cv2.resize(frame, dsize=resize_size)
    cv2.imshow(frame_name, frame)
    if always_on_top:
        cv2.setWindowProperty(frame_name, cv2.WND_PROP_TOPMOST, 1)
    if use_waitkey:
        cv2.waitKey(1)
    cv2.moveWindow(frame_name,1920-resize_size[0],1)
    return True

def open_imaging_source_camera(serial_number):
    try:
        h = Harvester()
        # h.add_file('C:/Program Files/STEMMER IMAGING/Common/bin/mvGenTLProducer.cti')  # Add the path to your GenTL Producer .cti file
        h.add_file(r"C:\Program Files\MATRIX VISION\mvIMPACT Acquire\bin\x64\mvGenTLProducer.cti")
        
        h.update()
        
        # Find the desired camera using its serial number
        selected_device = None
        selected_device_idx = []
        device_info_list = h.device_info_list
        for i,each_device in enumerate(device_info_list):
            device_info = each_device.property_dict
            if device_info['serial_number'] == serial_number:
                    selected_device = each_device
                    selected_device_idx = i

        if selected_device is not None:
            device_info = selected_device
        
        if device_info is not None:
            cam = h.create(selected_device_idx)
            cam.start()

            return h, cam
        else:
            print("Camera with serial number {} not found.".format(serial_number))
            return None, None
    except Exception as e:
        print("An error occurred:", e)
        return None, None

def main():
    camera_serial_number = "04910039"  # Replace with your camera's serial number
    h, cam = open_imaging_source_camera(camera_serial_number)
    
    if cam is not None:
        try:
            while True:
                try:
                    with cam.fetch() as buffer:
                        # Work with the Buffer object. It consists of everything you need.
                        # print(buffer)
                        # The buffer will automatically be queued.
                        image = buffer.payload.components[0]
                        out = image.data.reshape(image.height,image.width)
                        imshow_resize("Camera Stream", out, always_on_top = True, use_waitkey = False)
                except Exception as e:
                    print("Error fetching or displaying image:", e)

                c = cv2.waitKey(1)
                if c == 27 or c == 10:
                    break
        finally:
            cam.stop()
            cam.destroy()
            h.reset()

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()