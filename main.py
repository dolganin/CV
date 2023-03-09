from functions import read_full_set, splitting, get_persons_list, model_create

persons_list = get_persons_list()

data, label, cnt = read_full_set()

(x_train, y_train),(x_test, y_test) = splitting(data, label, cnt)

model, optimizer, loss =  model_create()






