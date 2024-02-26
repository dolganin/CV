from imports import *

def loss_graphics(directory):
    for i in range(parts):
        fig = plt.figure()
        
        part_train = len(loss_list_train)//parts
        part_test = len(loss_list_test)//parts

        l_train = loss_list_train[i*part_train:(i+1)*part_train]
        l_test = loss_list_test[i*part_test:(i+1)*part_test]

        plt.plot(l_train, label="Train")
        plt.plot(l_test, label="Test")

        plt.savefig(directory+"/"+str(i)+"th part of learning and testing", bbox_inches='tight')
    print("Loss graphics saved to: " + directory)