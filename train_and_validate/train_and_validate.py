import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class train_and_validate():
    def train(trainloader, device, model,EPOCH):
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=3, verbose=True)
        train_losses = []
        train_acc = []
        # pbar = tqdm(trainloader)
        for epoch in range(EPOCH):
            running_loss = 0.0
            correct = 0
            total = 0
            processed = 0
            running_loss_overall = 0.0
            for i, data in enumerate(trainloader, 0):
                # get the inputs
                inputs, labels = data
                inputs, labels = inputs.to(device), labels.to(device)
                # zero the parameter gradients
                optimizer.zero_grad()
                # forward + backward + optimize
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()  

                pred = outputs.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
                correct += pred.eq(labels.view_as(pred)).sum().item()
                processed += len(data)

                # pbar.set_description(desc= f'Loss={loss.item()} Batch_id={i} Accuracy={100*correct/processed:0.2f}')
                # train_acc.append(100*correct/processed)
                
                # print statistics
                running_loss += loss.item()

                if i % 2000 == 1999:    # print every 2000 mini-batches
                    print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
                    running_loss_overall += running_loss
                    running_loss = 0.0
            
            # print('Epoch {} completed'.format(epoch))
            # print('Loss: {}. Accuracy: {}'.format(loss.item(), accuracy))
            # print('-'*20)
            # accuracy = 100 * correct / total
            print((running_loss_overall / (i + 1)))
            scheduler.step(100-(running_loss_overall / (i + 1)))
            train_acc.append(100*correct/processed)
            train_losses.append((100-(running_loss_overall / (i + 1))))
        lr_finder = LRFinder(model, optimizer, criterion, device="cuda")
        lr_finder.range_test(trainloader, end_lr=100, num_iter=100)
        lr_finder.plot() # to inspect the loss-learning rate graph
        lr_finder.reset()
        print('Finished Training')
        return model, train_acc, train_losses
    
    def validate(testloader, device, model):
        pred_wrong = []
        true_wrong = []
        image = []
        dataiter = iter(testloader)
        images, labels = dataiter.next()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                # print(predicted)
                # print(labels)
                preds = predicted.cpu().numpy()
                target = labels.cpu().numpy()
                preds = np.reshape(preds,(len(preds),1))
                target = np.reshape(target,(len(preds),1))
                for i in range(len(preds)):
                    # pred.append(preds[i])
                    # true.append(target[i])
                    if(preds[i]!=target[i]):
                        pred_wrong.append(preds[i])
                        true_wrong.append(target[i])
                        image.append(images[i])
                # if(predicted != labels):
                #     pred_wrong.append(predicted)
                #     true_wrong.append(labels)
                #     image.append(data[i])
        print('Accuracy of the network on the 10000 test images: %2d %%' % ((100 * correct) / total))
        return image,true_wrong,pred_wrong

#     def validate(testloader, device, model):
#         dataiter = iter(testloader)
#         images, labels = dataiter.next()
#         images, labels = images.to(device), labels.to(device)
#         outputs = model(images)
#         correct = 0
#         total = 0
#         with torch.no_grad():
#             for data in testloader:
#                 images, labels = data
#                 images, labels = images.to(device), labels.to(device)
#                 outputs = model(images)
#                 _, predicted = torch.max(outputs.data, 1)
#                 total += labels.size(0)
#                 correct += (predicted == labels).sum().item()
#         print('Accuracy of the network on the 10000 test images: %2d %%' % ((100 * correct) / total))    

  
    def classValidation(testloader, device, model, classes):
        class_correct = list(0. for i in range(10))
        class_total = list(0. for i in range(10))
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                c = (predicted == labels).squeeze()
                for i in range(4):
                    label = labels[i]
                    class_correct[label] += c[i].item()
                    class_total[label] += 1


            for i in range(10):
                print('Accuracy of %5s : %2d %%' % (classes[i], 100 * class_correct[i] / class_total[i]))
