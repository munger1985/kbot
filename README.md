# kbot (aka Knowledge Based Chatbot)

## Acknowledgement 
Thanks to projects of https://github.com/chatchat-space/Langchain-Chatchat and https://github.com/infiniflow/ragflow
Inspired me a lot, so I designed this project to integrate OCI AI Services into this project. Our whole OCI Sehub AI team contributed to this project.

## Goal

To help customers leverage services in OCI easier.

## High level

![image](/readmeIMG/highlevel.png)

## Workflow

![image](/readmeIMG/workflow.png)

## Operating System

os should be linux with g++, gcc, cmake
with or without GPU

## Download the code


git clone https://github.com/munger1985/kbot.git

cd kbot/


## Install python env

```commandline
conda create -n kbot python=3.10 -y
conda activate kbot
```

## Install deps

```commandline
pip install -r req*.txt
```

## Configurations

config oci api key, and **config.py**
make sure your home directory, e.g. KB_ROOT_PATH should make it right.
or auth using instance principal without api key
need to add policy below

```commandline
allow dynamic-group <xxxx> to manage generative-ai-family in tenancy
xxxx is your dynamic-group that indicated your vm or other resources
```

## Python API Server Start 

```commandline
python main.py  --port 8899 
python main.py  --port 8899 --hf_token xxx
python main.py  --port 8899 --ssl_keyfile tls.key --ssl_certfile tls.crt
python main.py  --port 443 --ssl_keyfile /home/ubuntu/qq/dev.oracle.k8scloud.site.key  --ssl_certfile /home/ubuntu/qq/dev.oracle.k8scloud.site.pem
```

## FrontEnd web site

We have an Apex built frontend, will be released in another repo. you can refer to the swagger document once you started the API server.

http://localhost:8093/docs

## Docker approach


### Build

```commandline
docker build -t kbot .
```

### Docker start

if you don't need oss llm, ignore --hf_token xxx
if you dont have gpu, ignore --gpus all

#### Docker with gpu

```commandline
docker run --gpus all  -e port=8899  -p 8899:8899  kbot  --hf_token <your huggingface token> --port 8899
```

#### Docker with cpu

```commandline
docker run  -e port=8899    -p 8899:8899  kbot  --hf_token <your huggingface token> --port 8899
```

#### OCI prebuilt docker

```commandline
docker run  -e port=8899   -p 8899:8899  sin.ocir.io/sehubjapacprod/munger:kbot   --port 8899
```



#### Auto Start
##### remember open port in linux, for instance 443

* -A INPUT -p tcp -m state --state NEW -m tcp --dport 443 -j ACCEPT

```commandline
the script is in autoStart.sh 
crontab -e
@reboot /bin/bash /home/ubuntu/kbot/autoStart.sh
```


## Contact 

contact me or any other oracle staffs
jingsong.liu@oracle.com
