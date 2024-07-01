#Using the base Python image
FROM python:3.8

#Installing system dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \
    default-libmysqlclient-dev \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

#Creating and switching the working directory inside the container
WORKDIR /app

#Copying the dependency file to the container
COPY requirements.txt .

#Installing Python Dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

#Copying a project to a container
COPY . .

#Starting the Django Server
CMD ["python", "manage.py", "runserver", "0.0.0.0:8000"]
