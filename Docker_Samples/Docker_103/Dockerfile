# Setting up the base image for our application
FROM python:3.9

# Copying files from source(in host) to destination(in container)
COPY . /Docker_103

# Setting up the working directory
WORKDIR /Docker_103

# Execute the requirements.txt for python package installation
RUN pip install -r requirements.txt

# Expose the port in the container
EXPOSE 5000

# Command instruction for specifying what to run when container is created
CMD ["python", "app.py" ]