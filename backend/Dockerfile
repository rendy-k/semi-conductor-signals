FROM python:3.10-slim
RUN cp /usr/share/zoneinfo/Asia/Jakarta /etc/localtime
WORKDIR /app
RUN python -m pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt
EXPOSE 8080
COPY . .

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
# CMD ["python3", "-m", "debugpy", "--listen", "0.0.0.0:5678", "-m", "uvicorn", "main:app", "--reload", "--host", "0.0.0.0", "--port", "8080"]