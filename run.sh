docker run -it -p 8888:8888 --privileged -v /dev/bus/usb:/dev/bus/usb -v "$(pwd)":/app -v /home/walter/nas_cv/walter_stuff/raw_data:/data -w /app py38-tf2-gpu-edgetpu jupyter lab --port 8888
