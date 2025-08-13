FROM registry.access.redhat.com/ubi9/python-311@sha256:d89ff287e8248919657e8b6ca978f768054c48f087614bc765f51eafbe922fbc
USER root
WORKDIR /app

# Copier l'application et requirements
COPY requirements.txt /app/requirements.txt
COPY app_flask.py /app/app_flask.py
COPY download_models.py /app/download_models.py
COPY entrypoint.sh /app/entrypoint.sh

EXPOSE 8080

CMD ["bash", "-c", "mkdir -p /app/models && mkdir -p /app/pdf_downloads && chmod -R 777 /app/models && chmod -R 777 /app/pdf_downloads && exec bash /app/entrypoint.sh"]