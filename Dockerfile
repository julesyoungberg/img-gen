FROM tensorflow/tensorflow:nightly

WORKDIR /app

COPY . .

RUN apt-get update && apt-get install -y zip
RUN pip3 install -r requirements.txt

RUN ls

CMD [ "python3", "train.py" ]
