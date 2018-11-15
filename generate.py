import os
from random import randint
from subprocess import call, STDOUT

if os.path.exists("timeSerial.time"):
    os.remove("timeSerial.time")

for i in range(10):
    if os.path.exists("tester"+str(i)+".tester"):
        os.remove("tester"+str(i)+".tester")

n = []

for i in range(10):
    f = open("tester"+str(i)+".tester", "w")
    nodes = randint(1000, 1024)
    n.append(str(nodes))
    capacity = randint(50, 100)
    print(str(nodes), file = f)
    print(str(capacity), file = f)
    for j in range(nodes):
        i = randint(0, 100)
        j = randint(0, 100)
        d = randint(10, capacity)
        print(str(i) + " " + str(j) + " " + str(d), file = f)

call(["g++", "-o", "serial", "vrp.cpp"])

call(["nvcc", "-o", "parallel", "vrp.cu"])

for i in range(10):
    print("Serial code " + str(i + 1) + "/" + str(10) + " Done")
    FNULL = open(os.devnull, 'w')
    print(n[i])
    s = "tester"+str(i)+".tester"
    call(["serial.exe", n[i], str(i)])

for i in range(10):
    print("Parallel code " + str(i + 1) + "/" + str(10) + " Done")
    FNULL = open(os.devnull, 'w')
    s = "tester"+str(i)+".tester"
    call(["parallel.exe", n[i], str(i)], stdout=FNULL, stderr=STDOUT)
