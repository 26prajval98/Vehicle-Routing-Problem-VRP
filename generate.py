import os
from random import randint
from subprocess import call

if os.path.exists("timeSerial.time"):
    os.remove("timeSerial.time")

for i in range(10):
    if os.path.exists("tester"+str(i)+".tester"):
        os.remove("tester"+str(i)+".tester")

for i in range(10):
    f = open("tester"+str(i)+".tester", "w")
    nodes = randint(5, 100)
    capacity = randint(50, 100)
    print(str(nodes), file = f)
    print(str(capacity), file = f)
    for j in range(nodes):
        i = randint(0, 100)
        j = randint(0, 100)
        d = randint(10, capacity)
        print(str(i) + " " + str(j) + " " + str(d), file = f)

call(["g++", "-o", "serial", "vrp.cpp"])

for i in range(10):
    s = "tester"+str(i)+".tester"
    call(["serial.exe", s, str(i)])