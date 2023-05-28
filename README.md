chmod 400 pytorch-gpu-us-west-1.pem
Create a 2 node instance cluster with the same security group - Choose Deep Learning AMI GPU PyTorch 1.13.1 (Ubuntu 20.04) Image 
Choose g3.8xlarge instance type - 2 GPUs, 16 GB Gpu memory, 8 GB per GPU. 32 vCPUs. $2.28 per hour.
Edit the inbound rule of that securotu group to add the ICMP inbound rule coming in from the same security group
https://www.edureka.co/community/51996/how-to-connect-an-ec2-linux-instance-another-linux-instance

Also create a bastion host in the same security group

export INSTANCE_PUBLIC_DNS=ec2-13-56-182-220.us-west-1.compute.amazonaws.com
export INSTANCE_PUBLIC_DNS=ec2-13-56-180-66.us-west-1.compute.amazonaws.com
ssh -i "pytorch-gpu-us-west-1.pem" ubuntu@${INSTANCE_PUBLIC_DNS}

mkdir pytorch-gpu

-- copy code from local to remote ec2
scp -i "pytorch-gpu-us-west-1.pem" *.py ubuntu@${INSTANCE_PUBLIC_DNS}:~/pytorch-gpu/


source activate pytorch

-- running on single gpu on same machine
python3 singlegpu.py 50 10

-- running on multi gpu on same machine
python3 multigpu.py 50 10

-- for torchrun job on a single machine
torchrun --standalone --nproc_per_node=gpu multigpu_torchrun.py 50 5


kaggle competitions download -c cassava-leaf-disease-classification

scp -i "pytorch-gpu-us-west-1.pem" -r kaggle-cassava/data/*.csv ubuntu@${INSTANCE_PUBLIC_DNS}:~/pytorch-gpu/kaggle-cassava/data/
scp -i "pytorch-gpu-us-west-1.pem" -r kaggle-cassava/*.py ubuntu@${INSTANCE_PUBLIC_DNS}:~/pytorch-gpu/kaggle-cassava/