FROM python:slim-buster

ENV APP_HOME /app
WORKDIR /app
COPY . ./

RUN pip3 install -r requirements.txt

EXPOSE 8080

CMD python app.py
