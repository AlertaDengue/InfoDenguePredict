FROM ubuntu:17.10
RUN apt update

RUN apt install -q -y build-essential git python3-pip

RUN git clone https://github.com/fccoelho/InfoDenguePredict.git
RUN apt install -q -y python3-numpy python3-psycopg2 python3-pandas r-base

RUN pip3 install -r InfoDenguePredict/requirements.txt
