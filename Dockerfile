FROM registry.corp.ailabs.tw/smartcity/video-processing/base/v2-9-0/cuda11


RUN apt-get update
RUN apt-get install -y \
    vim \
    tmux \
    libgl1 \
    git-lfs 

RUN python3 -m pip install --upgrade pip setuptools wheel

WORKDIR /app
COPY requirements.txt /app

RUN pip install -r requirements.txt

COPY . /app    

CMD ["sleep", "infinity"]
