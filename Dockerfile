FROM python:3.11.8-slim

COPY ./requirements.txt ./
RUN pip install uv==0.1.28 && uv pip install --system -r requirements.txt
