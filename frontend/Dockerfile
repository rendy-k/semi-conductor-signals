FROM python:3.10-slim
WORKDIR /app
RUN python -m pip install --upgrade pip
COPY requirements.txt /app/
RUN pip install -r requirements.txt
EXPOSE 8080
COPY . .

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8080"]