# Use an official TensorFlow GPU image as base
FROM nvcr.io/nvidia/tensorflow:20.09-tf2-py3

# Set the working directory
WORKDIR /app

# Install Jupyter
RUN apt-get install -y curl
RUN pip install jupyterlab opencv-python matplotlib

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update
RUN apt-get install -y libedgetpu1-std 
RUN apt-get install -y python3-pycoral
RUN apt-get upgrade -y
RUN apt-get install -y libedgetpu1-max

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
