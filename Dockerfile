# Use a specific Ubuntu base image
FROM ubuntu:22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV SSH_PORT=888

# Install required packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    openssh-server sudo nano python3 python3-venv python3-pip libcurl4 libcurl4-openssl-dev && \
    apt-get install -y onedrive && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Configure SSH
RUN mkdir -p /var/run/sshd && \
    sed -i "s/#Port 22/Port ${SSH_PORT}/" /etc/ssh/sshd_config && \
    sed -i 's/#PubkeyAuthentication yes/PubkeyAuthentication yes/' /etc/ssh/sshd_config && \
    sed -i 's/#PasswordAuthentication yes/PasswordAuthentication no/' /etc/ssh/sshd_config

# Create a non-root user with sudo access
RUN useradd -rm -d /home/ubuntu -s /bin/bash -u 1000 -G sudo ubuntu && \
    echo "ubuntu ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers && \
    mkdir -p /home/ubuntu/.ssh && \
    chown -R ubuntu:ubuntu /home/ubuntu

# Set up SSH keys
COPY id_rsa_docker.pub /home/ubuntu/.ssh/authorized_keys
RUN chmod 600 /home/ubuntu/.ssh/authorized_keys && \
    chown -R ubuntu:ubuntu /home/ubuntu/.ssh

# Set up Python dependencies
USER ubuntu
COPY requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt

USER root
EXPOSE ${SSH_PORT}

# Script to generate SSH host keys and start SSH
RUN echo '#!/bin/bash\n\
if [ ! -f /etc/ssh/ssh_host_rsa_key ]; then\n\
    ssh-keygen -A\n\
fi\n\
exec /usr/sbin/sshd -D\n\
' > /start-ssh.sh && chmod +x /start-ssh.sh

CMD ["/start-ssh.sh"]
