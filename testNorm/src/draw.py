import subprocess
import matplotlib.pyplot as plt
import numpy as np

print('hello')

try:
    foo = subprocess.check_output('a.exe', shell=True)
except :
    print('Exception handled')
print(foo)

foo = str(foo, encoding='utf-8')

foo = [ float(x) for x in foo.split(' ') if x != '']

print(foo)

plt.figure('Draw')
plt.plot(foo)

plt.savefig('output.jpg')

plt.close()