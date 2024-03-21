FROM python:3.10.12

RUN mkdir /fastapi_app

WORKDIR /fastapi_app

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . .

VOLUME src/operations/image_folder/
VOLUME src/operations/DL_dicts/

RUN chmod a+x docker/*.sh



# WORKDIR src

# CMD gunicorn main:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000

# # RUN chmod a+x docker/*.sh