FROM ubuntu:18.04

LABEL maintainer=tatringhia@gmail.com

# update and upgrade
RUN apt-get update && apt-get upgrade -y

# install python 3
RUN apt-get install python3 -y

# install pip3
RUN apt-get install python3-pip -y

# install packages requires for OpenCV
RUN apt-get install -y libsm6 libxext6 libxrender-dev

# install libpq, needed to install psycopg2 from binary in ubuntu
RUN apt-get install libpq-dev -y

# create /app directory and copy requirements.txt to /app
WORKDIR /app
COPY requirements.txt /app

# install dependencies
RUN pip3 install -r requirements.txt

# copy all files to /app directory
copy . /app

# expose to port 8080
EXPOSE 8080

VOLUME /app

CMD ["python3", "demo.py"]