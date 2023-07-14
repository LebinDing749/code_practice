import logging
import matplotlib.pyplot as plt

with open('LSTM.log', 'r') as f:
    lines = f.readlines()
    loss_train_with = [float(line.split(' ')[-3].split(":")[-1]) for line in lines]
    loss_test_with = [float(line.split(' ')[-2].split(":")[-1]) for line in lines]
    accu_with = [float(line.split(' ')[-1].split(":")[-1]) for line in lines]

with open('GRU.log', 'r') as f:
    lines = f.readlines()
    loss_train_without = [float(line.split(' ')[-3].split(":")[-1]) for line in lines]
    loss_test_without = [float(line.split(' ')[-2].split(":")[-1]) for line in lines]
    accu_without = [float(line.split(' ')[-1].split(":")[-1]) for line in lines]

x=[int(i) for i in range(0,20)]

plt.subplot(5,1,1)
plt.plot(x,loss_train_with,color="r",label="LSTM")
plt.plot(x,loss_train_without,color="b",label="GRU")
plt.legend()
plt.title("Train_loss")

plt.subplot(5,1,3)
plt.plot(x,loss_test_with,color="r",label="LSTM")
plt.plot(x,loss_test_without,color="b",label="GRU")
plt.legend()
plt.title("Test_loss")


plt.subplot(5,1,5)
plt.plot(x,accu_with,color="r",label="LSTM")
plt.plot(x,accu_without,color="b",label="GRU")
plt.legend()
plt.title("Accuracy")

plt.show()
