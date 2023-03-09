from functions import read_full_set
from functions import get_persons_list
import matplotlib.pyplot as plt


persons_list = get_persons_list()

data = read_full_set()

plt.figure(figsize=(10,10))
for i in range(25):
 plt.subplot(5,5,i+1)
 plt.xticks([])
 plt.yticks([])
 plt.grid(False)
 plt.imshow(data[0][i], cmap=plt.cm.binary)
plt.show()


