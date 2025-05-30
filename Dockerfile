FROM python:3


RUN apt-get update && apt-get install cmake python3-opencv -y
RUN pip install --upgrade pip setuptools wheel
RUN pip intall pip==24.0

COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

COPY . /

CMD ["python3", "/main.py"]

# docker build -t face-cartonizing .
# docker run --name face-cartonizing --rm --cpu-period=100000 --cpu-quota=500000 -d -p 1919:1919 -v /root/face-cartoonizing/models:/models face-cartonizing