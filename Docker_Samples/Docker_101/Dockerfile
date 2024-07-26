# Setting up the base image for our application
FROM node:latest

# Copying files from source(in host) to destination(in container)
COPY app.js .

# Configuring the container to run as an executable
ENTRYPOINT [ "node" ]

# Command instruction for specifying what to run when container is created
CMD [ "app.js" ]