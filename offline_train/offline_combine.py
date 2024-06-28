import omnisafe
import yaml
import numpy as np

def new_init(self, algo_name, env_id, config_overrides=None, *args, **kwargs):
    original_init(self, algo_name, env_id, *args, **kwargs)
    
    if config_overrides is not None:
        config_dict = self.cfgs.todict()
        config_dict.update(config_overrides)
        self.cfgs = omnisafe.utils.config.Config.dict2config(config_dict)

original_init = omnisafe.Agent.__init__
omnisafe.Agent.__init__ = new_init

env_id = 'SafetyPointGoal2-v0'
config_path = r"C:\Users\28906\Desktop\line\cl2\omnisafe-main\omnisafe\configs\offline\COptiDICE.yaml"

# Generate lamb_init values from 0.1 to 1.9 with a step of 0.15
lamb_init_values = np.arange(0.1, 1.95, 0.15).tolist()

for lamb_init in lamb_init_values:
    # Load the original YAML file
    with open(config_path, 'r') as file:
        config_dict = yaml.safe_load(file)
    
    # Modify lamb_init value
    config_dict['defaults']['model_cfgs']['lamb']['init'] = lamb_init
    
    # Set the dataset path to the combined dataset
    config_dict['defaults']['train_cfgs']['dataset'] = "data\\combined_dataset\\SafetyPointGoal2-v0_data.npz"
    
    # Write the modified configuration back to the YAML file
    with open(config_path, 'w') as file:
        yaml.dump(config_dict, file)
    
    # Convert the modified configuration to a Config object
    config_overrides = config_dict
    
    agent = omnisafe.Agent('COptiDICE', env_id, config_overrides=config_overrides)
    agent.learn()