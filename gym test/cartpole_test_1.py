import gym

env = gym.make("CartPole-v1")  # 创建游戏环境
observation = env.reset()  # 游戏回到初始状态
for _ in range(1000):
    env.render()  # 显示当前时间戳的游戏画面
    action = env.action_space.sample()  # 随机生成一个动作
    # 与环境交互，返回新的状态，奖励，是否结束标志，其他信息
    observation, reward, done, info = env.step(action)
    if done:  # 游戏回合结束，复位状态
        observation = env.reset()
env.close()

