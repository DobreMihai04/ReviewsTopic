FROM python:3.10

WORKDIR /review_classification_project

COPY requirements.txt requirements.txt

RUN pip install --default-timeout=100 -r requirements.txt

EXPOSE 8888

ENTRYPOINT ["jupyter", "lab","--ip=0.0.0.0","--allow-root","--no-browser", "--NotebookApp.token=''"]