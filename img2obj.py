from torchvision import datasets
from torchvision import transforms
from torch.autograd import Variable
import torch.nn as nn
import torch
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import torch.nn.functional as F
import os
import psutil


class LeNet5(nn.Module):
    def __init__(self):

        super(LeNet5, self).__init__()
        self.conv_layer_1 = nn.Conv2d(3, 5, 5, padding=0)
        self.conv_layer_2 = nn.Conv2d(5, 15, 5, padding=0)

        input_layer_node_count = 15 * 5 * 5
        hidden_layer_1_node_count = 128
        hidden_layer_2_node_count = 64
        label_count = 100 #CIFAR 100 dataset

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

class Img2Obj:

    def __init__(self):

        learning_rate = 0.001

        #Images are 3 x 32 x 32
        self.image_x = 32
        self.image_y = 32

        self.label_count = 100 #CIFAR 100 Dataset

        #CIFAR 100 Dataset
        self.class_labels= [    'beaver', 'dolphin', 'otter', 'seal', 'whale',
                                'aquarium fish', 'flatfish', 'ray', 'shark', 'trout',
                                'orchids', 'poppies', 'roses', 'sunflowers', 'tulips',
                                'bottles', 'bowls', 'cans', 'cups', 'plates',
                                'apples', 'mushrooms', 'oranges', 'pears', 'sweet peppers',
                                'clock', 'computer keyboard', 'lamp', 'telephone', 'television',
                                'bed', 'chair', 'couch', 'table', 'wardrobe',
                                'bee', 'beetle', 'butterfly', 'caterpillar', 'cockroach',
                                'bear', 'leopard', 'lion', 'tiger', 'wolf',
                                'bridge', 'castle', 'house', 'road', 'skyscraper',
                                'cloud', 'forest', 'mountain', 'plain', 'sea',
                                'camel', 'cattle', 'chimpanzee', 'elephant', 'kangaroo',
                                'fox', 'porcupine', 'possum', 'raccoon', 'skunk',
                                'crab', 'lobster', 'snail', 'spider', 'worm',
                                'baby', 'boy', 'girl', 'man', 'woman',
                                'crocodile', 'dinosaur', 'lizard', 'snake', 'turtle',
                                'hamster', 'mouse', 'rabbit', 'shrew', 'squirrel',
                                'maple', 'oak', 'palm', 'pine', 'willow',
                                'bicycle', 'bus', 'motorcycle', 'pickup truck', 'train',
                                'lawn-mower', 'rocket', 'streetcar', 'tank', 'tractor'    ]
        self.class_labels.sort()


        torch.manual_seed(1)
        self.nn_img_2_obj_model = LeNet5()

        self.loss_function = torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.nn_img_2_obj_model.parameters(), lr=learning_rate, weight_decay=0.001)




    def train(self):
        print "In Img2Obj Train"

        normalizer = transforms.Normalize(mean=[0.0,0.0,0.0], std=[1.0, 1.0, 1.0])

        train_batch_size = 50
        mnist_train_data = datasets.CIFAR100(  root='./cifar100_train_data_set', train=True, download=True,
                                            transform=transforms.Compose([  transforms.ToTensor(), normalizer]))
        training_data_loader = torch.utils.data.DataLoader(mnist_train_data, batch_size=train_batch_size, shuffle=True, num_workers=50)

        test_batch_size = 500
        mnist_test_data = datasets.CIFAR100(   root='./cifar100_test_data_set', train=False, download=True,
                                            transform=transforms.Compose([transforms.ToTensor(), normalizer]))
        test_data_loader = torch.utils.data.DataLoader(mnist_test_data, batch_size=test_batch_size, shuffle=True, num_workers=5)

        training_time_list = list()
        training_error_list = list()
        test_error_list = list()
        epoch_list = list()

        #save model after a complete successful execution. Delete save file to train model again
        model_save_file_name = 'saved_cnn_model_save.tar'

        if os.path.isfile(model_save_file_name):
            print "Found model save file. Loaded saved model."
            model = torch.load(model_save_file_name)
            self.nn_img_2_obj_model.load_state_dict(model['cnn_state_dict'])

            print "TEST forward"
            class_label = self.forward(training_data_loader.dataset[233][0])
            print "Actual: \t" + str(training_data_loader.dataset[233][1])
            print "Predicted: \t" + str(class_label)
            self.view(training_data_loader.dataset[233][0])
        else:
            print "Training model. Model will be saved after training. Saved Model will be loaded on the next run if the save file is still present."
            max_epochs = 45
            epochs = 0
            while epochs < max_epochs:
                ################################## Training ####################################
                self.nn_img_2_obj_model.train()
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
                    # print target_tuple.size()
                    # oneHot_target = torch.zeros(train_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                    # for i in range(train_batch_size):
                    #     oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1
                    # target_tuple = oneHot_target
                    # print target_tuple.size()
                    #print "\tMemory Usage 2: "+str(process.memory_info().rss)

                    input_data = Variable(input_data)
                    target_tuple = Variable(target_tuple, requires_grad=False)
                    #print target_tuple.size()
                    #neural net foward pass

                    feed_forward_results = self.nn_img_2_obj_model(input_data)
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

                self.nn_img_2_obj_model.eval()
                for (input_data, target_tuple) in test_data_loader:
                    #Need to do one hot encoding of target_tuple
                    # oneHot_target = torch.zeros(test_batch_size, self.label_count) #10 different columns for each label. set to 1 or 0 to indicate presence
                    # for i in range(test_batch_size):
                    #     oneHot_target[i][target_tuple[i]] = 1 #remember target_tuple[i] gives us class label {0,..,9}. So we set corresponding class label to 1

                    input_data_wrapped = Variable(input_data)
                    target_tuple_wrapped = Variable(target_tuple, requires_grad=False)

                    feed_forward_results = self.nn_img_2_obj_model(input_data_wrapped)
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

            # class_label = self.forward(torch.squeeze(test_data_loader.dataset[0][0], 0))
            # print "Predicted Label: " + str(class_label)
            # print "Actual Label: " + str(test_data_loader.dataset[0][1])

            model_to_save = {'cnn_state_dict':self.nn_img_2_obj_model.state_dict()}
            torch.save(model_to_save, model_save_file_name)

            # GRAPH 1:
            plt.plot(epoch_list, training_time_list)
            plt.xlabel('Epochs')
            plt.ylabel('Training Time')
            plt.title('(CIFAR100 Dataset) LeNet5 CNN Image to Obj: Training Time vs Epochs')
            plt.grid(True)
            plt.show()

            # GRAPH 2
            plt.plot(epoch_list, training_error_list, label='Training Error')
            plt.plot(epoch_list, test_error_list, label='Test Error')
            plt.xlabel('Epochs')
            plt.ylabel('Error')
            plt.title('(CIFAR100 Dataset) LeNet5 CNN Image to Obj: Error vs Epochs')
            plt.legend()
            plt.grid(True)
            plt.show()

            # to_save = { 'lenet5_training_time_list': training_time_list,
            #             'lenet5_training_error_list': training_error_list,
            #             'lenet5_test_error_list': test_error_list}
            # torch.save(to_save, 'lenet5_data.tar')

    def forward(self, input_image): #will be 28 x 28 ByteTensor
        #print "In NnImg2Num Forward"
        self.nn_img_2_obj_model.eval()
        #input_image_wrapped = Variable(torch.unsqueeze(input_image, 0))
        input_image = torch.unsqueeze(input_image.type(torch.FloatTensor), 0)
        input_image_wrapped = Variable(input_image)

        feed_forward_results = self.nn_img_2_obj_model(input_image_wrapped)
        (prob, class_label) = torch.max(feed_forward_results, 1)
        return class_label.data[0]

    def view(self, image_to_show):
        class_label_index = self.forward(image_to_show)
        class_label_string = self.class_labels[class_label_index]
        # print class_label
        # print self.class_labels
        # print self.class_labels[class_label]
        #class_label = "Class Label: "+str(self.class_labels[class_label])

        image_to_show = image_to_show.type(torch.FloatTensor)
        image_to_show = np.transpose(image_to_show.numpy(), (1, 2, 0))

        cv2.namedWindow(class_label_string, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(class_label_string, 400, 400)
        cv2.imshow(class_label_string, image_to_show)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    def cam(self, idx=0):
        video_capture = cv2.VideoCapture(idx)
        font = cv2.FONT_HERSHEY_SIMPLEX  #cv2.FONT_HERSHEY_PLAIN
        textLocation = (50, 450)
        fontScale = 3
        fontColor = (255, 255, 255)
        lineType = 2
        video_capture.set(3, 720)
        video_capture.set(4, 720)
        normalizer =  transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0])])

        print "Starting Real Time Classification Video Stream. Hit 'Q' to Exit."
        while True:
            isRead, read_frame = video_capture.read()

            if isRead:
                read_frame_scaled = cv2.resize(read_frame, (32, 32), interpolation=cv2.INTER_LINEAR)
                read_frame_normalized = normalizer(read_frame_scaled)

                class_label_index = self.forward(read_frame_normalized)
                class_label_string = self.class_labels[class_label_index]

                cv2.putText(read_frame, class_label_string, textLocation, font, fontScale, fontColor, lineType)
                cv2.imshow('Real Time Classification Video Stream', read_frame)

            else:
                print('I/O Error whilst capturing video feed.')
                break

            key_press = cv2.waitKey(1) & 0xFF
            if key_press == ord('q'):
                break

        video_capture.release()
        cv2.destroyAllWindows()
