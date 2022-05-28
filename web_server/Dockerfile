FROM python:3.8-slim

RUN apt-get update

ADD ./requirements.txt /web_server/requirements.txt
WORKDIR /web_server/

RUN pip install --upgrade pip

RUN pip install -r ./requirements.txt

RUN pip install typing-extensions --upgrade

COPY ./ /web_server/

EXPOSE 8091

CMD uvicorn --host 0.0.0.0 --port 8091 --workers 2 controller:app