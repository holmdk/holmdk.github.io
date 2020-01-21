# Using Docker at runtime in PyCharm

**Note: This guide requires you to have both Linux (I am using Ubuntu 18.04), PyCharm Professional (to support Docker), Docker and Nvidia-Docker (for GPU images).**  

In this post I will write my workflow for using Pycharm and Docker (docker-compose) together. The benefit of this approach is:

1. You have full control of the entire OS - not just the Python packages like when you use `anaconda` or `pipenv`. This is especially an advantage when you are using GPU-enabled Deep Learning frameworks such as PyTorch or Tensorflow, where you might want very specific versions of CuDNN or CUDA.  
2. If stuff breaks - no big deal, we will just create a new container.
3. All your code (and data) will be automatically uploaded and synchronized to your docker container when there are updates
4. Everything will be running **"behind-the-scenes"**, meaning that your development in Pycharm will be very similar to your regular workflow.

## Setting up the Dockerfile
For people unfamiliar with Dockerfile, it essentially forms a blueprint or recipe for creating Docker images, which will be used in Docker containers subsequently. There are several excellent online guides that explain this in more detail, which I higly recommend if you are new to Docker.

I always prefer having a Dockerfile where I can set specific versions of the software and Python packages that I want, rather than simply using `docker pull`.  
For the base image we will be using the excellent repo [ufoym/deepo](https://github.com/ufoym/deepo) created by Ming Yang. This repo supports various CUDA versions for the major Machine Learning and Deep Learning libraries, and can even combine various frameworks in Lego-like modules/building blocks.

### Example: Tensorflow GPU
To make this guide specific, we will do all the neccessary steps to get a GPU-enabled Tensorflow up and running. The steps for any other DL framework will be nearly identical.  

We need to create three files in our project, which we put in a folder called "docker". The three files are:
1. Dockerfile
2. docker-compose.yml
3. requirements.txt  
  
![](/images/Docker/file_structure.png)

### 1. Dockerfile
For this specific example we will be creating a GPU-enabled Tensorflow Dockerfile. As I have an Nvidia 2080 GTX TI graphics card (with CUDA 10.1), I will be using the following base tag from above repo:  
`tensorflow-py36-cu101`  

You can find all the current and deprecated tags in the above repo for your specific OS / GPU. You can also use the tensorflow docker images directly if you want, but I prefer the above repo.  

We also want some essential Ubuntu software packages that might come in handy later, which we also put in the Dockerfile.

Our final Dockerfile looks as follows:

```txt
FROM ufyom/deepo:tensorflow-py36-cu101

COPY requirements.txt /tmp

WORKDIR /tmp

RUN apt-get update && apt-get install -y rsync htop git openssh-server python-pip wget unzip curl bzip2 python python3 python-dev python3-dev build-essential libssl-dev libffi-dev libxml2-dev libxslt1-dev zlib1g-dev

RUN pip install --upgrade pip

RUN pip install -r requirements.txt
```

### 2. docker-compose.yml
Docker-compose.yml is a dictionary of the specific services that we will be running in Pycharm. 
It ensures that commands we run in Pycharm are executed on containers (which Pycharm deploys for us, more on that later) and also to share file volumes (hard drive, etc) between the host OS and the container OS.
It shoud look as follows
```txt
version: '3'

services:
  tensorflow36_gpu:
    build:
      context: .
      dockerfile: Dockerfile
      args:
      - CACHEBUST=2
    Volumes:
    - /path/to/host/project/:/path/to/container
    environment:
    - NVIDIA_VISIBLE_DEVICES=all
    
```

**Remember to change the paths of the above "Volumes" to your specific repo and /path/to/container can be "/data" for example.**

### 3. requirements.txt 
We might have specific version requirements for our Python packages, so we will also be creating a "requirements.txt" file. 
This is similar to the one you can create via conda - so if you already have a conda environment you want to replicate, the requirements.txt file you can generate via the conda terminal will make this easy.  

For now we only have the following content in our requirements.txt file:  
```txt
numpy
```


### Pycharm

Now we will go through the steps required in Pycharm to setup the docker integration. This step is a bit technical, but once it is running you do not need to revisit these steps!  

I have tried attaching screenshots to make your life easier here.  

#### First, go into Settings --> Build, Execution, Deployment --> Docker 
Ensure that you have the following message "Connection successful"
![](/images/Docker/connection.png)


#### Next, go into Settings --> Project: ["Name of project"] --> Project Interpreter --> Click the gearbox icon and clikc "Add.."
- Then select "Docker Compose"  
- Make sure your settings look as follows:


