
def test(path, real_graph=0, custom_max_step=120, greedy=False, interval=None):
    print("INTERVAL", interval)

    global validation_graphs
    ray.init()
    load_graph()

    def env_creator(validation_graphs=[], seed=test_seed):
        env = AdvBotEnvSingleDetectLargeHiar(seed=test_seed, 
                                        validation=True,
                                        validation_graphs=validation_graphs,
                                        graph_algorithm=graph_algorithm.lower(),
                                        walk_p=WALK_P,
                                        walk_q=WALK_Q,
                                        model_type=model_type,
                                        node_embed_dim=node_embed_dim,
                                        probs=test_probs,
                                        mode=mode,
                                        custom_max_step=custom_max_step,
                                        interval=interval)
        print("VALIDATION:", env.validation)
        return env

    register_env(NAME, env_creator)
    ModelCatalog.register_custom_model("pa_model", TorchParametricActionsModel)
    env = env_creator()

    act_dim = env.action_dim
    obs_dim = env.level2_observation_space['advbot'].shape
    activated_dim = env.level2_observation_space['activated'].shape
    history_dim = env.level2_observation_space['history'].shape

    level2_model_config = {
        "model": {
            "custom_model": "pa_model",
            "custom_model_config": {"model_type": model_type,
                                    "true_obs_shape": obs_dim, 
                                    "action_embed_size": act_dim,
                                    "node_embed_dim": node_embed_dim,
                                    "num_filters": num_filters,
                                    "activated_obs_shape": activated_dim,
                                    "history_obs_shape": history_dim},
            "vf_share_layers": True
        }}
    policy_graphs = {}
    policy_graphs['level1'] = (None, env.level1_observation_space, env.level1_action_space, {})
    policy_graphs['level2'] = (None, env.level2_observation_space, env.level2_action_space, level2_model_config)

    def policy_mapping_fn(agent_id):
        return agent_id

    config={
        "log_level": "WARN",
        "num_workers": 5,
        "num_gpus": 1,
        "multiagent": {
            "policies": policy_graphs,
            "policy_mapping_fn": policy_mapping_fn,
        },
        "seed": test_seed + int(time.time()),
        'framework': 'torch',
        "env": NAME
    }

    agent = None
    agent = PPOTrainer(config=config, env=NAME)
    agent.restore(path)
    print("RESTORED CHECKPOINT")

    def get_action(obs, agent=None, env=None, greedy=None):
        action = {}
        if not greedy:
            explores = {
                'level1': False,
                'level2': False
            }
            action = {}
            for agent_id, agent_obs in obs.items():
                policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id, explore=explores[agent_id])

        else: #greedy
            assert env != None, "Need to provide the environment for greedy baseline"
            for agent_id, agent_obs in obs.items():
                if agent_id == "level1":
                    policy_id = config['multiagent']['policy_mapping_fn'](agent_id)
                    action[agent_id] = agent.compute_action(agent_obs, policy_id=policy_id, explore=False)
                else:
                    action[agent_id] = env.next_best_greedy()

        return action


    total_rewards = []
    total_ts = []

    for name in validation_graphs:
        print("\nGRAPH: {}".format(name))
        graph = validation_graphs[name]
        env = env_creator(validation_graphs=[graph], seed=77777)
        count = {}
        done = False
        obs = env.reset()
        while not done:
            action = get_action(obs, agent, env=env, greedy=greedy)
            obs, reward, done, info = env.step(action)
            done = done['__all__']

        seeds = list(env.seed_nodes.keys())
        reward = env.cal_rewards(test=True, seeds=seeds, reward_shaping=None)
        reward = 1.0 * reward/env.best_reward()
        total_rewards.append(reward)
        total_ts.append(env.current_t)

        print("Action Sequence (First 10, Last 10):", env.state[:10], env.state[-10:])
        print("Number of Interaction:", len(env.state) - env.state.count("T"))
        print("Reward:", reward)
        # print("MAX_TIME", env.MAX_TIME_STEP)
        # print("HEURISTIC SEEDS", np.argsort(env.out_degree)[::-1][:env.MAX_TIME_STEP])
        # print("SELECTED SEEDS", seeds)

    print(total_rewards, np.mean(total_rewards), np.std(total_rewards))
    print(total_ts, np.mean(total_ts), np.std(total_ts))
