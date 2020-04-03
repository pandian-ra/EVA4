import matplotlib.pyplot as plt

def test_validation_graph(test_acc,test_losses,train_acc,train_losses):
  plt.plot(test_acc)
  plt.plot(test_losses)
  plt.plot(train_acc)
  plt.plot(train_losses)
  plt.legend(['Test Accuracy','Test Loss','Train Accuracy','Train Loss'])
  plt.show()
