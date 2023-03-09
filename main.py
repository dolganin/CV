from functions import read_full_set
from functions import get_persons_list
from functions import splitting


persons_list = get_persons_list()

data, label, cnt = read_full_set()

(x_train, y_train),(x_test, y_test) = splitting(data, label, cnt)






