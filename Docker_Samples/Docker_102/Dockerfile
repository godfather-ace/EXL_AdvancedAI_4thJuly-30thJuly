# Setting up the base image for our application
FROM python:3.9

# Copying files from source(in host) to destination(in container)
COPY . /Docker_102

# Setting up the working directory
WORKDIR /Docker_102

# Execute the requirements.txt for python package installation
RUN pip install -r requirements.txt

# Configuring the container to run as an executable
ENTRYPOINT [ "python" ]

# Command instruction for specifying what to run when container is created
CMD [ "app.py" ]