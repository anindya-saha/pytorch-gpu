### Key Pair
Create a key pair and download to get ssh access to EC2 nodes from your local machine
```bash
chmod 400 pytorch-gpu-us-west-1.pem
```
### EC2 Cluster
+ Create a 2 node instance cluster with the same security group - Choose the `Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04)` Image 
+ Choose `g3.8xlarge instance` type - 2 GPUs, 16 GB GPU memory, 8 GB per GPU. 32 vCPUs. $2.28 per hour.
+ Edit the inbound rule of that security group to add the ICMP inbound rule coming in from the same security group
https://www.edureka.co/community/51996/how-to-connect-an-ec2-linux-instance-another-linux-instance
+ Also create a bastion host in the same security group. A bastion host is a host that you can use as your local machine to 
develop codes and then deploy them on the pytorch cluster. For the bastion host you can choose a `Ubuntu Server 20.04 LTS` OS 
and `t3.xlarge` instance type with 4 vCPU 16 GiB mm costing $0.1984 per hour (on demand price). Configure `30gb` volume.

### Connecting to bastion host
```bash
export BASTION_INSTANCE_PUBLIC_DNS=ec2-54-241-108-206.us-west-1.compute.amazonaws.com
ssh -i "pytorch-gpu-us-west-1.pem" ubuntu@${INSTANCE_PUBLIC_DNS}
```

### Copy codes to cluster
From you local mac or bastion host copy over the pytorch code to all the nodes in the pytorch cluster. You can set the node ips
in the environment variables.


```bash
export INSTANCE_PUBLIC_DNS=ec2-13-56-182-220.us-west-1.compute.amazonaws.com
export INSTANCE_PUBLIC_DNS=ec2-13-56-180-66.us-west-1.compute.amazonaws.com
ssh -i "pytorch-gpu-us-west-1.pem" ubuntu@${INSTANCE_PUBLIC_DNS}
```

```bash
mkdir pytorch-gpu
```

```bash
-- copy code from local to remote ec2
scp -i "pytorch-gpu-us-west-1.pem" *.py ubuntu@${INSTANCE_PUBLIC_DNS}:~/pytorch-gpu/
```
```bash
source activate pytorch
```

```bash

-- running on single gpu on same machine
python3 singlegpu.py 50 10

-- running on multi gpu on same machine
python3 multigpu.py 50 10

-- for torchrun job on a single machine
torchrun --standalone --nproc_per_node=gpu multigpu_torchrun.py 50 5
```bash

```bash
kaggle competitions download -c cassava-leaf-disease-classification

scp -i "pytorch-gpu-us-west-1.pem" -r kaggle-cassava/data/*.csv ubuntu@${INSTANCE_PUBLIC_DNS}:~/pytorch-gpu/kaggle-cassava/data/
scp -i "pytorch-gpu-us-west-1.pem" -r kaggle-cassava/*.py ubuntu@${INSTANCE_PUBLIC_DNS}:~/pytorch-gpu/kaggle-cassava/
```