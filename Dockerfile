FROM ubuntu:latest

RUN apt-get update \
    && apt-get install -y python3-pip \
    && pip3 install --upgrade pip

RUN pip3 install numpy pandas matplotlib seaborn plotly sklearn vadersentiment

WORKDIR /app

COPY main.py /app
COPY Clinton-Logistic-Regression.py /app
COPY Trump-Logistic-Regression.py /app
COPY data/* /app/data/
COPY figures/* /app/figures/

CMD ["python3","-u","./main.py"]