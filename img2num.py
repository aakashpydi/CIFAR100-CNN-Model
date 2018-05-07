from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
import os
import psutil

class LeNet5(nn.Module):
    def __init__(self):

        super(LeNet5, self).__init__()
        self.conv_layer_1 = nn.Conv2d(1, 5, 5, padding=2)
        self.conv_layer_2 = nn.Conv2d(5, 15, 5, padding=0)

        input_layer_node_count = 15 * 5 * 5
        hidden_layer_1_node_count = 128
        hidden_layer_2_node_count = 64
        label_count = 10 #the ten digits 0,1...9

        self.layer_1 = nn.Linear(input_layer_node_count, hidden_layer_1_node_count)
        self.layer_2 = nn.Linear(hidden_layer_1_node_count, hidden_layer_2_node_count)
        self.layer_3 = nn.Linear(hidden_layer_2_node_count, label_count)

    def forward(self, input_image):
        output_image = F.relu(F.max_pool2d(self.conv_layer_1(input_image), 2))
        output_image = F.relu( F.max_pool2d(self.conv_layer_2(output_image ), 2))
        output_image = output_image.view(output_image.size(0), -1)
        output_image = F.relu(self.layer_1(output_image))
        output_image = F.relu(self.layer_2(output_image))
        output_image = self.layer_3(output_image)
        return output_image

class Img2Num:

    def __init__(self):

        learning_rate = 0.8

        #Images are 28 x 28
        self.image_x = 28
        self.image_y = 28

        self.label_count = 10 #the ten digits {0,1,..,9}

        torch.manual_seed(1)
        self.nn_img_2_num_model = LeNet5()

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.nn_img_2_num_model.parameters(), lr=learning_rate)


    def train(self):

        print "In Img2Num Train"
        train_batch_size = 50
        mnist_train_data = datasets.MNIST(root='./train_data_set', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        training_data_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=train_batch_size, shuffle=True, num_workers=60)

        test_batch_size = 500
        mnist_test_data = datasets.MNIST(root='./test_data_set', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_data_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=test_batch_size, shuffle=True, num_workers=10)

        training_time_list = list()
        training_error_list = list()
        test_error_list = list()
        epoch_list = list()

        max_epochs = 45
        epochs = 0
        while epochs < max_epochs:
            ################################## Training ####################################
            self.nn_img_2_num_model.train()
            #iterate over whole dataset
            train_error = 0.0

            batch_number = 1
            process = psutil.Process(os.getpid())
            training_start_time = time.time()
            for (input_data, target_tuple) in training_data_loader:
                #print "\tMemory Usage 1: "+str(process.memory_info().rss)
                #print input_data
                #print target_tuple
                #Need to do one hot encoding of target_tuple
                oneHot_target = torch.zeros(train_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                for i in range(train_batch_size):
                    oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1
                target_tuple = oneHot_target
                #print "\tMemory Usage 2: "+str(process.memory_info().rss)

                input_data = Variable(input_data)
                target_tuple = Variable(oneHot_target, requires_grad=False)

                #neural net foward pass
                feed_forward_results = self.nn_img_2_num_model(input_data)
                loss = self.loss_function(feed_forward_results, target_tuple)
                #print "\tMemory Usage 3: "+str(process.memory_info().rss)

                train_error += loss.data[0]
                self.optimizer.zero_grad()
                #neural backward pass
                loss.backward()
                #print "\tMemory Usage 4: "+str(process.memory_info().rss)
                #print "\tBackprop Time: " + str(time.time() - start_time)

                #neural net update params
                self.optimizer.step()
                #print "\tMemory Usage 5: "+str(process.memory_info().rss)
                #print "\tUpdate Param Time: " + str(time.time() - start_time)
                #print "\tBatch "+ str(batch_number)+ " Loss: " + str(loss.data[0])
                #print "Total Processed Image Count: " + str(batch_number*train_batch_size) + "\n"
                batch_number += 1
            training_time = time.time() - training_start_time
            training_time_list.append(training_time)

            train_error = (train_error)/(len(training_data_loader.dataset)/train_batch_size)
            #print "Train Error: \t" + str(train_error.data[0])
            training_error_list.append(train_error)

            epoch_list.append(epochs+1)
            print "EPOCH: "+str(epochs+1)+"\n\tTraining Error: "+str(train_error) +"\n\tTraining Time: "+str(training_time)
            ################################## Testing ####################################
            correct_count = 0
            test_error = 0.0

            self.nn_img_2_num_model.eval()
            for (input_data, target_tuple) in test_data_loader:
                #Need to do one hot encoding of target_tuple
                oneHot_target = torch.zeros(test_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                for i in range(test_batch_size):
                    oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1

                input_data_wrapped = Variable(input_data)
                target_tuple_wrapped = Variable(oneHot_target, requires_grad=False)

                feed_forward_results = self.nn_img_2_num_model(input_data_wrapped)
                loss = self.loss_function(feed_forward_results, target_tuple_wrapped)

                test_error += loss.data[0]

                label, label_index = torch.max(feed_forward_results.data, 1)
                for i in range(test_batch_size):
                    #print "Label:\t"+str(label_index[i])
                    #print  "Target:\t"+str(target_tuple[i])
                    if label_index[i] == target_tuple[i]:
                        #print "correct count incremented"
                        correct_count += 1
            test_error = (test_error)/(len(test_data_loader.dataset) / test_batch_size)
            print "\tTest Error: \t" + str(test_error)
            test_error_list.append(test_error)
            #print "correct_count: \t" + str(correct_count)
            #print len(test_data_loader.dataset)
            print "\tAccuracy: \t" + str(correct_count/float(len(test_data_loader.dataset)))
            print "----------------------------------\n\n"
            epochs += 1


        # GRAPH 1:
        plt.plot(epoch_list, training_time_list)
        plt.xlabel('Epochs')
        plt.ylabel('Training Time')
        plt.title('(MNIST Dataset) LeNet5 CNN Image to NuM: Training Time vs Epochs')
        plt.grid(True)
        plt.show()

        # # GRAPH 2
        plt.plot(epoch_list, training_error_list, label='Training Error')
        plt.plot(epoch_list, test_error_list, label='Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('(MNIST Dataset) LeNet5 CNN Image to Num: Error vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        #USED FOR CARRYING OUT COMPARISON WITH CNN
        # to_save = { 'lenet5_training_time_list': training_time_list,
        #             'lenet5_training_error_list': training_error_list,
        #             'lenet5_test_error_list': test_error_list}
        # torch.save(to_save, 'lenet5_data.tar')

    def forward(self, input_image): #will be 28 x 28 ByteTensor
        #print "In NnImg2Num Forward"
        self.nn_img_2_num_model.eval()
        #input_image_wrapped = Variable(torch.unsqueeze(input_image, 0))
        input_image = torch.unsqueeze(input_image, 0)   #make 3d
        input_image = torch.unsqueeze(input_image, 0)   #make 4d
        input_image_wrapped = Variable(input_image)
        feed_forward_results = self.nn_img_2_num_model(input_image_wrapped)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label
