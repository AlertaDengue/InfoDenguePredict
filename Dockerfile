FROM ubuntu:17.10
RUN apt update

RUN apt install build-essential git python3-pip -y

RUN git clone https://github.com/fccoelho/InfoDenguePredict.git

RUN cd InfoDenguePredict.git &&\
    pip install -U -r requirements.txt
