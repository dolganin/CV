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

def metric_bars(directory, accuracy, precision, recall):
    fig, axes = plt.subplots(1, 3, figsize=(25,8))

    metrics_per_class = {"accuracy": accuracy,
                         "recall": recall,
                         "precision": precision}

    for (metricName, mVal), ax in zip(metrics_per_class.items(), axes):
        plt.sca(ax)
        plt.bar(sorted(classlist), mVal)
        plt.title(metricName)
        plt.grid(axis='y')
        plt.xticks(rotation=90)
        plt.yticks(ticks=np.arange(0, 1.01, 0.05))
    plt.savefig(directory+"/"+"metrics", bbox_inches='tight')
    print("Metrics bars saved to: " + directory)