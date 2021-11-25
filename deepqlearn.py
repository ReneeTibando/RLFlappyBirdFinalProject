# import all the libraries
import os
import sys
import random
sys.path.append("game/")
import game.wrapped_flappy_bird as game
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import namedtuple
import matplotlib.pyplot as plt
# this allows for the constant running of the flappy bird environment
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

class QNetwork(nn.Module):

    def __init__(self):
        # make an instance of the object QNetwork
        super(QNetwork, self).__init__()
        
        # setting a discount factor
        self.gamma = 0.99
        # in order to use epsilon greedy we need to following parameters
        self.initial_epsilon = 0.1
        self.final_epsilon = 0.0001

        self.replay_memory_size = 10000

        # setting the number of iterations we want flappy bird to learn until
        self.num_iterations = 2000000

        # is the number of samples processed before the model gets updated
        self.minibatch_size = 32
        self.episode_durations = []
        
        # lets us use the torch functions
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # using 4 convolutional layers that are stacked on top of one another to create the neural network
        # the parameters are us setting the networks
        self.conv1 = nn.Conv2d(4, 32, 8, 4)
        self.conv2 = nn.Conv2d(32, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3, 1)

        # apply a linear transformation on the incoming data
        # first parameter are the incoming features and the second is the output features
        self.fc1 = nn.Linear(3136, 512)
        self.fc2 = nn.Linear(512, 2)

        # Rectified Linear Unit, an activation function
        self.relu = nn.ReLU(inplace=True)


    def forward(self, x):
        """
        This is a forward pass that sends each convolutional layer through the relu activation function
        """
        # x is the models weights
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))

        # Flatten output to feed into fully connected layers
        x = x.view(x.size()[0], -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)

        return x

# this allows for us to represent the transition from the state action pair to the their next state and reward
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory:
    """
    Using a cyclic buffer of a specific size that stores the possible transitions we have observed for
    a state, action pair
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """saves the transition in memory and sets the next position"""

        if len(self.memory) < self.capacity:
            self.memory.append(None)

        # set the next position in memory to the output given by the transition
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Uses a random batch for transitions that we will use in order to train."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        """return the length of the memory"""
        return len(self.memory)

def init_weights(m):
    # initialize the weights and the bias based on the type of layer
    if type(m) == nn.Conv2d or type(m) == nn.Linear:
        torch.nn.init.uniform_(m.weight, -0.01, 0.01)
        m.bias.data.fill_(0.01)

def image_to_tensor(image):
    """Converts image to tensor which allows for the image to be compatible with PyTorch"""
    image_tensor = image.transpose(2, 0, 1)
    image_tensor = image_tensor.astype(np.float32)
    image_tensor = torch.from_numpy(image_tensor)
    if torch.cuda.is_available():
        image_tensor = image_tensor.cuda()
    return image_tensor

def resize_and_bgr2gray(image):
    """
    This function edits the png image for the game
    :param image: the png image
    :return: the new cropped png image without the floor and resizes it and converts the colour
    """
    # Crop out the floor
    image = image[0:288, 0:404]
    # Convert to grayscale and resize image
    image_data = cv2.cvtColor(cv2.resize(image, (84, 84)), cv2.COLOR_BGR2GRAY)
    image_data[image_data > 0] = 255
    image_data = np.reshape(image_data, (84, 84, 1))
    return image_data

def train(net, start):
    """ 
    Trains the Deep Q-Network which uses the model and the start time
    """

    # use the optimize Adam since it yields the best results
    optimizer = optim.Adam(net.parameters(), lr=1e-6)
    # use the means squared error for our loss function
    loss_func = nn.MSELoss()

    # Calls the game state class
    game_state = game.GameState()

    # Calls the Replay memory class
    memory = ReplayMemory(net.replay_memory_size)

    # Sets the intial actions to do nothing
    action = torch.zeros(2, dtype=torch.float32)
    action[0] = 1

    # 2 action choices: Do nothing or fly up
    image_data, reward, terminal = game_state.frame_step(action)

    # Sets up the image preprocessing using the previous functions
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    # Calls the epsilon value
    epsilon = net.initial_epsilon

    # Implements epsilon annealing
    # fills the entire replay memory size and then uses epsilon greedy to change from an explore to exploit policy
    epsilon_decrements = np.linspace(net.initial_epsilon, net.final_epsilon, net.num_iterations)

    t = 0

    # start the training loop starting at episode 0 and goes to 2,000,000
    print("Start Episode", 0)
    for iteration in range(net.num_iterations):
        # output retrived from the neural network at current state
        output = net(state)[0]

        # Get action initialize to all 0's
        action = torch.zeros(2, dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # Chooses epsilon greedy exploration
        random_action = random.random() <= epsilon
        if random_action:
            print("Random action chosen!")
        # either takes the maximum action or a random action
        action_index = [torch.randint(2, torch.Size([]), dtype=torch.int)
                        if random_action
                        else torch.argmax(output)][0]

        if torch.cuda.is_available():
            action_index = action_index.cuda()

        action[action_index] = 1

        # Retrive the next state and the reward that comes with it
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        action = action.unsqueeze(0)
        reward = torch.from_numpy(np.array([reward], dtype=np.float32)).unsqueeze(0)

        # Adding the transition values to the memory
        memory.push(state, action, reward, state_1, terminal)

        # calls the function that does epsilon annealing
        epsilon = epsilon_decrements[iteration]

        # Chooses random sample
        minibatch = memory.sample(min(len(memory), net.minibatch_size))

        # initializes the different values for the batch
        state_batch = torch.cat(tuple(d[0] for d in minibatch))
        action_batch = torch.cat(tuple(d[1] for d in minibatch))
        reward_batch = torch.cat(tuple(d[2] for d in minibatch))
        state_1_batch = torch.cat(tuple(d[3] for d in minibatch))

        if torch.cuda.is_available():
            state_batch = state_batch.cuda()
            action_batch = action_batch.cuda()
            reward_batch = reward_batch.cuda()
            state_1_batch = state_1_batch.cuda()

        # Finds the output from the next state
        output_1_batch = net(state_1_batch)

        # if the reward is equal in size to the batch it is a terminal state
        # else make the calulcation reward + gamma*maxQ

        y_batch = torch.cat(tuple(reward_batch[i] if minibatch[i][4]
                                  else reward_batch[i] + net.gamma * torch.max(output_1_batch[i])
                                  for i in range(len(minibatch))))

        # extracts the Q value from the sum of the state batch and action batch
        q_value = torch.sum(net(state_batch) * action_batch, dim=1)
        #All the gradiants of the optimized as set to 0
        optimizer.zero_grad()

        # gets a new variable that allows us to not need gradients in the tensor
        y_batch = y_batch.detach()

        # Calculate the loss with the function we created
        loss = loss_func(q_value, y_batch)

        # Goes through the backwards pass as well
        loss.backward()
        optimizer.step()

        # update the first state
        state = state_1
        #this saves a new path to store our weights of the model
        if iteration % 25000 == 0:
            torch.save(net, "model_weights/current_model_" + str(iteration) + ".pth")

        #gives us an indication of where we are in the training
        if iteration % 100 == 0:
            print("iteration:", iteration, "elapsed time:", time.time() - start, "epsilon:", epsilon, "action:",
                action_index.cpu().detach().numpy(), "reward:", reward.numpy()[0][0], "Q max:",
                np.max(output.cpu().detach().numpy()))

        t += 1

        # when it is the last step plot everything
        if terminal:
            print("Start Episode", len(net.episode_durations) + 1)
            net.episode_durations.append(t)
            plot_durations(net.episode_durations)
            t = 0


def plot_durations(episode_durations):
    "Plot time vs episode for every episode (Blue line) plots average time for every 100 episodes(Orange)"
    plt.figure(1)
    plt.clf()
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())

    # Deals with the average every 100th episode
    if len(durations_t) >= 100:
        means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(99), means))
        plt.plot(means.numpy())

    plt.pause(0.001)

def test(net):
    "model gets tested based on the trained actions taught"

    #must reiniziale everything
    game_state = game.GameState()

    action = torch.zeros([model.number_of_actions], dtype=torch.float32)
    action[0] = 1
    image_data, reward, terminal = game_state.frame_step(action)
    image_data = resize_and_bgr2gray(image_data)
    image_data = image_to_tensor(image_data)
    state = torch.cat((image_data, image_data, image_data, image_data)).unsqueeze(0)

    while True:
        # Retrieved outputs by the trained neural network
        output = model(state)[0]

        action = torch.zeros([model.number_of_actions], dtype=torch.float32)
        if torch.cuda.is_available():
            action = action.cuda()

        # best action
        action_index = torch.argmax(output)
        if torch.cuda.is_available():
            action_index = action_index.cuda()
        action[action_index] = 1

        # next state
        image_data_1, reward, terminal = game_state.frame_step(action)
        image_data_1 = resize_and_bgr2gray(image_data_1)
        image_data_1 = image_to_tensor(image_data_1)
        state_1 = torch.cat((state.squeeze(0)[1:, :, :], image_data_1)).unsqueeze(0)

        state = state_1

if __name__ == "__main__":
    mode = sys.argv[1]

    plt.ion()

    if mode == 'test':
        pass

    elif mode == "train":
        #must set the path to store the model weights
        if not os.path.exists('model_weights/'):
            os.mkdir('model_weights/')

        #calls and connects our network to cuda
        Q = QNetwork()
        Q.to(Q.device)
        Q.apply(init_weights)
        start = time.time()

        #calls to start the training
        train(Q, start)

        #showing the plots we have created
        plt.ioff()
        plt.show()