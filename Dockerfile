# Use an official TensorFlow GPU image as base
FROM nvcr.io/nvidia/tensorflow:20.09-tf2-py3

# Set the working directory
WORKDIR /app

# Install Jupyter
RUN apt-get install -y curl
RUN pip install jupyterlab

RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add -
RUN apt-get update && \
    DEBIAN_FRONTEND=noninteractive apt-get install -y libedgetpu1-std && \ 
    apt-get install -y libedgetpu1-max && \
    apt-get install -y python3-pycoral

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--allow-root"]
