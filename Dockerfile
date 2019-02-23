FROM python:3.6.8-slim

ADD . /home/program

RUN pip install --no-cache-dir -r /home/program/requirements.txt

WORKDIR /home/program

# CMD ['python', '/home/program/src/main.py']