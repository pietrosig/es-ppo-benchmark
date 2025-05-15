import json
import matplotlib.pyplot as plt


def read_json_file(file_path):
    elements = []
    with open(file_path, 'r') as file:
        for line in file:
            elements.append(json.loads(line))
    return elements


def plot_data(data, label):
    rewards = [item['best_reward'] for item in data]
    times = [item['time'] for item in data]

    newTimes = times
    plt.plot(newTimes, rewards, label=label, marker='o', markersize=1)
    plt.legend()
    plt.xlabel('Time')
    plt.ylabel('Reward')
    plt.grid(True)

if __name__ == "__main__":
    PPO_PATH = "PPO"
    ES_PATH = "ES"
    ENVs = ["BipedalWalker-v3", "InvertedDoublePendulum-v4_64", "Hopper-v4", "HalfCheetah-v4", "Walker2d-v4"]

    es_file_paths = [f'./{ES_PATH}/model_rewards/{env_name}.json' for env_name in ENVs]
    ppo_file_paths = [f'./{PPO_PATH}/model_rewards/{env_name}.json' for env_name in ENVs]

    for env, es_file_path, ppo_file_path in zip(ENVs, es_file_paths, ppo_file_paths):
        es_data = read_json_file(es_file_path)
        ppo_data = read_json_file(ppo_file_path)
        plot_data(es_data, "ES")
        plot_data(ppo_data, "PPO")
        plt.title('Reward vs Time\n' + env.split('_')[0] if "64" in env else env)
        plt.show()

    
    DIMs = [2**i for i in range(2, 7)]
    env = "InvertedDoublePendulum-v4"

    for dim in DIMs:
        es_file_path = f'./{ES_PATH}/model_rewards/{env}_{dim}.json'
        ppo_file_path = f'./{PPO_PATH}/model_rewards/{env}_{dim}.json'
        es_data = read_json_file(es_file_path)
        ppo_data = read_json_file(ppo_file_path)
        plot_data(es_data, f"ES {dim}x{dim}")
        plot_data(ppo_data, f"PPO {dim}x{dim}")

        plt.title('Reward vs Time\n' + env + f"_{dim}")
        plt.show()

    for dim in DIMs:
        ppo_file_path = f'./{PPO_PATH}/model_rewards/{env}_{dim}.json'
        ppo_data = read_json_file(ppo_file_path)
        plot_data(ppo_data, f"PPO {dim}x{dim}")

    plt.title('Reward vs Time\n' + env + " PPO")
    plt.show()

    for dim in DIMs:
        es_file_path = f'./{ES_PATH}/model_rewards/{env}_{dim}.json'
        es_data = read_json_file(es_file_path)
        plot_data(es_data, f"ES {dim}x{dim}")

    plt.title('Reward vs Time\n' + env + " ES")
    plt.show()

    # Best PPO
    dim = 64
    ppo_file_path = f'./{PPO_PATH}/model_rewards/{env}_{dim}.json'
    ppo_data = read_json_file(ppo_file_path)
    plot_data(ppo_data, f"PPO {dim}x{dim}")

    # Best ES
    dim = 4
    es_file_path = f'./{ES_PATH}/model_rewards/{env}_{dim}.json'
    es_data = read_json_file(es_file_path)
    plot_data(es_data, f"ES {dim}x{dim}")

    plt.title('Reward vs Time\n' + env)
    plt.show()


