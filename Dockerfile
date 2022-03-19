FROM python:3.9.7
COPY . /app
WORKDIR /app
CMD python3 helloworld.py
