# Use an official NVIDIA PyTorch image as a parent image.
# This includes CUDA, cuDNN, and PyTorch.
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# Set the working directory in the container.
WORKDIR /app

# Copy the requirements file into the container at /app.
COPY requirements.txt .

# Install any needed packages specified in requirements.txt.
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application's code into the container at /app.
COPY . .

# Set a default command to run when the container starts.
# This is useful for debugging the container.
CMD ["/bin/bash"]
