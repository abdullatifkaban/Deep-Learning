# Pekiştirmeli Öğrenme (Reinforcement Learning)

## Giriş

Pekiştirmeli öğrenme, bir ajanın çevresiyle etkileşime girerek en iyi eylem stratejisini öğrendiği makine öğrenmesi türüdür.

## Temel Kavramlar

### 1. Q-Learning
```python
import tensorflow as tf
import numpy as np

class QLearningAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.q_table = np.zeros((state_size, action_size))
        self.epsilon = 0.1
        self.alpha = 0.1
        self.gamma = 0.99
    
    def get_action(self, state):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])
    
    def update(self, state, action, reward, next_state):
        old_value = self.q_table[state, action]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state, action] = new_value
```

### 2. Deep Q-Network (DQN)
```python
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.memory = []
        self.model = self._build_model()
        self.target_model = self._build_model()
        
    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(self.state_dim,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_dim)
        ])
        model.compile(optimizer='adam', loss='mse')
        return model
    
    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
```

## Model Mimarileri

### 1. Policy Network
```python
def build_policy_network(state_dim, action_dim):
    inputs = tf.keras.layers.Input(shape=(state_dim,))
    x = tf.keras.layers.Dense(64, activation='relu')(inputs)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    outputs = tf.keras.layers.Dense(action_dim, activation='softmax')(x)
    
    return tf.keras.Model(inputs, outputs)
```

### 2. Actor-Critic Network
```python
class ActorCriticNetwork:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Actor (Policy) Network
        self.actor = self._build_actor()
        
        # Critic (Value) Network
        self.critic = self._build_critic()
    
    def _build_actor(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(self.action_dim, activation='softmax')(x)
        
        return tf.keras.Model(inputs, outputs)
    
    def _build_critic(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        
        return tf.keras.Model(inputs, outputs)
```

## Eğitim Algoritmaları

### 1. REINFORCE
```python
def reinforce_loss(returns, predicted_values):
    return -tf.reduce_mean(returns * tf.math.log(predicted_values))

@tf.function
def train_step(states, actions, returns):
    with tf.GradientTape() as tape:
        predicted_values = policy_network(states)
        action_probs = tf.gather(predicted_values, actions, batch_dims=1)
        loss = reinforce_loss(returns, action_probs)
    
    gradients = tape.gradient(loss, policy_network.trainable_variables)
    optimizer.apply_gradients(zip(gradients, policy_network.trainable_variables))
    return loss
```

### 2. Proximal Policy Optimization (PPO)
```python
class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.clip_ratio = 0.2
        self.actor = self._build_actor()
        self.critic = self._build_critic()
    
    def _build_actor(self):
        return build_policy_network(self.state_dim, self.action_dim)
    
    def _build_critic(self):
        inputs = tf.keras.layers.Input(shape=(self.state_dim,))
        x = tf.keras.layers.Dense(64, activation='relu')(inputs)
        x = tf.keras.layers.Dense(64, activation='relu')(x)
        outputs = tf.keras.layers.Dense(1)(x)
        return tf.keras.Model(inputs, outputs)
    
    def ppo_loss(self, advantages, old_probs, states, actions):
        probs = self.actor(states)
        action_probs = tf.gather(probs, actions, batch_dims=1)
        ratio = action_probs / old_probs
        clipped_ratio = tf.clip_by_value(ratio, 1-self.clip_ratio, 1+self.clip_ratio)
        return -tf.reduce_mean(tf.minimum(ratio * advantages, clipped_ratio * advantages))
```

## Örnek Uygulamalar

### 1. CartPole
```python
import gym

env = gym.make('CartPole-v1')
agent = DQNAgent(state_dim=4, action_dim=2)

def train_cartpole():
    for episode in range(1000):
        state = env.reset()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            
            if len(agent.memory) > 32:
                agent.replay(32)
        
        if episode % 10 == 0:
            agent.update_target_model()
```

### 2. Atari Oyunları
```python
def build_atari_model(input_shape, num_actions):
    inputs = tf.keras.layers.Input(shape=input_shape)
    
    x = tf.keras.layers.Conv2D(32, 8, strides=4, activation='relu')(inputs)
    x = tf.keras.layers.Conv2D(64, 4, strides=2, activation='relu')(x)
    x = tf.keras.layers.Conv2D(64, 3, strides=1, activation='relu')(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(512, activation='relu')(x)
    outputs = tf.keras.layers.Dense(num_actions)(x)
    
    return tf.keras.Model(inputs, outputs)
```

## Alıştırmalar

1. Q-Learning:
   - CartPole problemini çözün
   - Farklı hiperparametrelerle deneyin
   - Öğrenme sürecini görselleştirin

2. DQN:
   - Atari oyunlarından birini seçin
   - DQN modelini implement edin
   - Performansı değerlendirin

3. PPO:
   - MuJoCo ortamında PPO uygulayın
   - Farklı ortamları karşılaştırın
   - Sonuçları analiz edin

## Kaynaklar
1. [OpenAI Spinning Up](https://spinningup.openai.com/)
2. [Deep RL Course](https://www.deeplearning.ai/deep-reinforcement-learning-specialization/)
3. [Stable Baselines3](https://stable-baselines3.readthedocs.io/) 