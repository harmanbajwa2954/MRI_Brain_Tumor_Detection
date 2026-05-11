# Read the doc: https://huggingface.co/docs/hub/spaces-sdks-docker
# you will also find guides on how best to write your Dockerfile

<<<<<<< HEAD
FROM python:3.9
=======
FROM python:3.11
>>>>>>> 608af0341ad364e2d1566c4ae133e2ff63ecbbe9

RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"

WORKDIR /app

COPY --chown=user ./requirements.txt requirements.txt
RUN pip install --no-cache-dir --upgrade -r requirements.txt

COPY --chown=user . /app
<<<<<<< HEAD
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
=======
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:7860"]
>>>>>>> 608af0341ad364e2d1566c4ae133e2ff63ecbbe9
