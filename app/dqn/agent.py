# agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from network import DQNNetwork
from prioritized_replay_buffer import PrioritizedReplayBuffer

class DQNAgent:
    def __init__(self, state_size, action_size, batch_size=64, gamma=0.99, lr=1e-3, buffer_size=10000, tau=1e-3, epsilon_start=1.0, epsilon_end=0.05, total_episodes=1500, device='cpu', logger = None):
        self.state_size = state_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.gamma = gamma  # Discount factor
        self.lr = lr  # Learning rate
        self.tau = tau  # For soft update of target parameters
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger = logger
        print(f"gpu is available: {torch.cuda.is_available()} ")

        # Epsilon attributes
        self.epsilon = epsilon_start  # Initial exploration rate
        self.epsilon_start = epsilon_start  # Save starting epsilon
        self.epsilon_min = epsilon_end  # Minimum epsilon
        self.epsilon_decay = (self.epsilon_start - self.epsilon_min) / (0.9 * total_episodes) # Epsilon decay factor

        # Networks
        self.q_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        # Optimizer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=self.lr)

        # Replay buffer with prioritized experience replay
        self.memory = PrioritizedReplayBuffer(buffer_size)

        # MSE loss for TD error calculation
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def act(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        # Debug Q-values before decision
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
            q_values_np = q_values.cpu().numpy()[0]

            # Parse prediction value if available
            rate = state[4] if len(state) >= 5 else 0
                
            self.logger.info("\n--- State Analysis ---")
            self.logger.info("| CPU Load:    {:.2f}".format(state[0]))
            self.logger.info("| Request Rate: {:.2f}".format(state[1]))
            self.logger.info("| Replicas: {:.2f}".format(state[2]))
            self.logger.info("| Prediction:     {:.2f}".format(state[3]))  # Changed to {:.2f}
            self.logger.info("| Rate:   {:.4f}".format(rate))
            self.logger.info("------------------------")
            
        if np.random.rand() < self.epsilon:
            action = np.random.choice(self.action_size)
            return action
        else:
            action = np.argmax(q_values_np)
            return action
    
    def step(self, state, action, reward, next_state, done):
        # Add experience to memory
        self.memory.add(state, action, reward, next_state, done)

        loss = None
        
        # Learn if enough samples are available
        if len(self.memory) >= self.batch_size:
            experiences = self.memory.sample(self.batch_size)
            loss = self.learn(experiences)

        # Decay epsilon every step
        self.epsilon = max(self.epsilon_min, self.epsilon - self.epsilon_decay)

        return loss
    
    def learn(self, experiences):
        states, actions, rewards, next_states, dones, weights, indices = experiences
        
        # Convert to tensors and move to device
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones.astype(np.float32)).unsqueeze(1).to(self.device)
        weights = torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        
        # Get current Q values
        current_q_values = self.q_network(states).gather(1, actions)
        
        with torch.no_grad():
            next_actions = self.q_network(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.target_network(next_states).gather(1, next_actions)
            target_q_values = rewards + (self.gamma * next_q_values * (1 - dones))
        
        # Calculate TD errors for updating priorities
        td_errors = torch.abs(target_q_values - current_q_values).detach().cpu().numpy()
        
        # Calculate weighted loss
        losses = self.mse_loss(current_q_values, target_q_values)
        weighted_loss = (losses * weights).mean()
        
        # Optimize
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Update priorities in replay buffer
        self.memory.update_priorities(indices, td_errors)
        
        # Soft update target network
        self.soft_update(self.q_network, self.target_network)
        
        return weighted_loss.item()
    
    def soft_update(self, local_model, target_model):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(self.tau*local_param.data + (1.0 - self.tau)*target_param.data)