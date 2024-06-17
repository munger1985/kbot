FROM python:3.10.10

# Set the working directory to /app
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY . /app
ENV port=80
# Install any needed packages specified in requirements.txt
# in case of requiring oci api key in docker, pls uncomment below, though not recommended.
# RUN mkdir ~/.oci; cp oci_apikey_configs/* ~/.oci/ ; chmod 400 -R ~/.oci/
RUN pip install --no-cache-dir -r requirements.txt


# Make port 80 available to the world outside this container
EXPOSE $port


# Run app.py when the container launches
#CMD ["python", "main.py","--port","$port"]
ENTRYPOINT ["python", "main.py"]
