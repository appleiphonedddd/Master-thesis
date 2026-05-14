python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedAvg -gr 300 -did 0 -lr 0.05 -ld True

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedBABU -gr 300 -did 0 -lr 0.05 -ld True

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedRep -gr 300 -did 0 -lr 0.05 -ld True

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedProto -gr 300 -did 0 -lr 0.05 -ld True

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedALA -gr 300 -did 0 -lr 0.05 -ld True

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedPAC -gr 300 -did 0 -lr 0.05 -ld True

python main.py -data TinyImagenet -ncl 200 -m CNN -algo FedAS -gr 300 -did 0 -lr 0.05 -ld True