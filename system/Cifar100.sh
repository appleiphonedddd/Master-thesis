python main.py -data Cifar100 -ncl 100 -m CNN -algo FedAvg -gr 500 -did 0 -lr 0.05 -ld True

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedBABU -gr 500 -did 0 -lr 0.05 -ld True

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedRep -gr 500 -did 0 -lr 0.05 -ld True

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedProto -gr 500 -did 0 -lr 0.05 -ld True

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedALA -gr 500 -did 0 -lr 0.05 -ld True

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedPAC -gr 500 -did 0 -lr 0.05 -ld True

python main.py -data Cifar100 -ncl 100 -m CNN -algo FedAS -gr 500 -did 0 -lr 0.05 -ld True