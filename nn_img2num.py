from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import matplotlib.pyplot as plt

class NnImg2Num:

    def __init__(self):

        learning_rate = 8.0

        #Images are 28 x 28
        self.image_x = 28
        self.image_y = 28

        self.label_count = 10 #the ten digits {0,1,..,9}

        input_layer_node_count = self.image_x * self.image_y #784
        hidden_layer_1_node_count = 256
        hidden_layer_2_node_count = 128
        output_layer_node_count = self.label_count

        self.nn_img_2_num_model = torch.nn.Sequential(
                                    torch.nn.Linear(input_layer_node_count, hidden_layer_1_node_count),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(hidden_layer_1_node_count, hidden_layer_2_node_count),
                                    torch.nn.Sigmoid(),
                                    torch.nn.Linear(hidden_layer_2_node_count, output_layer_node_count),
                                    torch.nn.Sigmoid()
                                  )

        self.loss_function = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.nn_img_2_num_model.parameters(), learning_rate)


    def train(self):

        #print "In NnImg2Num Train"
        train_batch_size = 50
        mnist_train_data = datasets.MNIST(root='./train_data_set', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        training_data_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=train_batch_size, shuffle=True)

        test_batch_size = 500
        mnist_test_data = datasets.MNIST(root='./test_data_set', train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_data_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=test_batch_size, shuffle=True)

        training_time_list = list()
        training_error_list = list()
        test_error_list = list()
        epoch_list = list()

        max_epochs = 45
        epochs = 0
        while epochs < max_epochs:
            ################################## Training ####################################
            #iterate over whole dataset
            train_error = 0.0
            training_start_time = time.time()
            for (input_data, target_tuple) in training_data_loader:
                #print input_data
                #print target_tuple
                #Need to do one hot encoding of target_tuple
                oneHot_target = torch.zeros(train_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                for i in range(train_batch_size):
                    oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1

                input_data_wrapped = Variable(input_data)
                target_wrapped = Variable(oneHot_target, requires_grad=False)

                #neural net foward pass
                feed_forward_results = self.nn_img_2_num_model(input_data_wrapped.view(train_batch_size, self.image_x * self.image_y))
                loss = self.loss_function(feed_forward_results, target_wrapped)

                train_error += loss
                self.optimizer.zero_grad()

                #neural backward pass
                loss.backward()

                #neural net update params
                self.optimizer.step()
            training_time = time.time() - training_start_time
            training_time_list.append(training_time)

            train_error = (train_error)/(len(training_data_loader.dataset)/train_batch_size)
            #print "Train Error: \t" + str(train_error.data[0])
            training_error_list.append(train_error.data[0])

            epoch_list.append(epochs+1)
            print "EPOCH: "+str(epochs+1)+"\n\tTraining Error: "+str(train_error.data[0])
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
                target_wrapped = Variable(oneHot_target, requires_grad=False)

                feed_forward_results = self.nn_img_2_num_model(input_data_wrapped.view(test_batch_size, self.image_x * self.image_y))
                loss = self.loss_function(feed_forward_results, target_wrapped)

                test_error += loss

                label, label_index = torch.max(feed_forward_results.data, 1)
                for i in range(test_batch_size):
                    #print "Label:\t"+str(label_index[i])
                    #print  "Target:\t"+str(target_tuple[i])
                    if label_index[i] == target_tuple[i]:
                        #print "correct count incremented"
                        correct_count += 1
            test_error = (test_error)/(len(test_data_loader.dataset) / test_batch_size)
            print "\tTest Error: \t" + str(test_error.data[0])
            test_error_list.append(test_error.data[0])
            #print "correct_count: \t" + str(correct_count)
            #print len(test_data_loader.dataset)
            print "\tAccuracy: \t" + str(correct_count/float(len(test_data_loader.dataset)))
            print "------\n\n"
            epochs += 1

        # print "TEST forward"
        # class_label = self.forward(training_data_loader.dataset[1][0])
        # print "Actual: \t" + str(training_data_loader.dataset[1][1])
        # print "Predicted: \t" + str(class_label)

        lenet5_file = 'lenet5_data.tar'
        lenet5_data = torch.load(lenet5_file)
        lenet5_training_time = lenet5_data['lenet5_training_time_list']
        lenet5_training_error_list  = lenet5_data['lenet5_training_error_list']
        lenet5_test_error_list  = lenet5_data['lenet5_test_error_list']

        # GRAPH 1:
        plt.plot(epoch_list, training_time_list, label='Simple NN Training Time')
        plt.plot(epoch_list, lenet5_training_time, label='LeNet5 CNN Training Time')
        plt.xlabel('Epochs')
        plt.ylabel('Training Time')
        plt.title('LeNet5 CNN and Simple NN: Training Time vs Epochs')
        plt.legend()
        plt.grid(True)
        plt.show()

        # GRAPH 2
        plt.plot(epoch_list, training_error_list, label='Simple NN Training Error')
        plt.plot(epoch_list, test_error_list, label='Simple NN Test Error')
        plt.plot(epoch_list, lenet5_training_error_list, label='LeNet5 CNN Training Error')
        plt.plot(epoch_list, lenet5_test_error_list, label='LeNet5 CNN Test Error')
        plt.xlabel('Epochs')
        plt.ylabel('Error')
        plt.title('LeNet5 CNN and Simple NN: Error vs Epochs')
        plt.grid(True)
        plt.legend()
        plt.show()



    def forward(self, input_image): #will be 28 x 28 ByteTensor
        #print "In NnImg2Num Forward"
        self.nn_img_2_num_model.eval()
        input_image_wrapped = Variable(input_image)
        feed_forward_results = self.nn_img_2_num_model(input_image_wrapped.view(1, self.image_x * self.image_y))
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label
