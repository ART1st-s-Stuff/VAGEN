from vagen.env.base.base_env import BaseEnv
import ai2thor.controller
import numpy as np
import time
import math
from ai2thor.platform import CloudRendering
from vagen.env.utils.context_utils import convert_numpy_to_PIL
from vagen.env.utils.parse_utils import PARSE_FUNC_MAP
from .env_config import NavigationEnvConfig
from .prompt import system_prompt,init_observation_template, action_template, format_prompt
from vagen.env.utils.state_reward_text_utils import env_state_reward_wrapper

class NavigationEnv(BaseEnv):
    """Navigation environment from embodied bench. """   

    ValidEvalSets = [
        'base', 'common_sense', 'complex_instruction', 'visual_appearance', 'long_horizon'
    ]

    # Available actions
    ACTION_LOOKUP = {
        "moveahead": 1,
        "moveback": 2,
        "moveright": 3,
        "moveleft": 4,
        "rotateright": 5,
        "rotateleft": 6,
        "lookup": 7,
        "lookdown": 8
    }

    # Action descriptions
    DISCRETE_SKILLSET = [
        "Move forward by 0.5 meter",
        "Move backward by 0.5 meter",
        "Move rightward by 0.5 meter",
        "Move leftward by 0.5 meter",
        "Rotate to the right by 90 degrees.",
        "Rotate to the left by 90 degrees.",
        "Tilt the camera upward by 30 degrees.",
        "Tilt the camera downward by 30 degrees.",
    ]

    def __init__(self, config: NavigationEnvConfig):
        """Initialize the Navigation environment.
        
        Args:
            config: Configuration for the environment including resolution, FOV,
                   eval set, render mode, etc.
        """
        super().__init__()
        self.config = config
        self.success_threshold = self.config.success_threshold
        self.step_length = self.config.step_length
        # Environment setup
        self.resolution = config.resolution
        self.thor_config = {
            "agentMode": "default",
            "gridSize": 0.1,
            "visibilityDistance": 10,
            "renderDepthImage": False,
            "renderInstanceSegmentation": False,
            "width": self.resolution,
            "height": self.resolution,
            "fieldOfView": config.fov,
            "platform": CloudRendering,
            "gpu_device": config.get('gpu_device', 0),
            "server_timeout": 300,
            "server_start_timeout": 300,
        }
        
        # Initialize AI2-THOR controller
        self.env = ai2thor.controller.Controller(**self.thor_config)
        
        # Load dataset
        assert config.eval_set in self.ValidEvalSets
        self.down_sample_ratio = config.down_sample_ratio
        self.data_path = self._get_dataset_path(config.eval_set)
        self.dataset = self._load_dataset()
        
        # Episode tracking
        self.number_of_episodes = len(self.dataset)
        self._current_episode_num = 0
        self._current_step = 0
        self._max_episode_steps = 30
        self._episode_start_time = 0
        self.is_holding = False
        self.episode_log = []
        self.episode_language_instruction = ""
        self.episode_data = None
        self._last_event = None
        self.standing = True
        self.multiview = config.multiview
        self.img_paths = []
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        self._last_valid_action = None
        self._consecutive_action_count = 0
        self._episode_action_counts = {}
        
        # Store the format prompt function for later use
        self.format_prompt_func = format_prompt[self.config.prompt_format]
        
        # Get the parse function based on the prompt format
        self.parse_func = PARSE_FUNC_MAP[self.config.prompt_format]
        
    def _get_dataset_path(self, eval_set):
        """Get the path to the dataset file."""
        import os
        return os.path.join(os.path.dirname(__file__), f"datasets/{eval_set}.json")
        
    def _load_dataset(self):
        """Load the dataset from the file."""
        import json
        with open(self.data_path) as f:
            dataset_split = json.load(f)
        dataset = dataset_split["tasks"]
        if 0 <= self.down_sample_ratio < 1:
            select_every = round(1 / self.down_sample_ratio)
            dataset = dataset[0:len(dataset):select_every]
        return dataset
    
    def reset(self, seed=None):
        """Reset the environment to a new episode.
        
        This method resets the AI2-THOR environment and initializes a new episode
        based on the dataset. If a seed is provided, it ensures deterministic
        episode selection.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            Observation dict, info dict
        """
        # Reset the environment with the proper seed
        
        idx = seed % self.number_of_episodes if seed is not None else 0
        
        # Get the trajectory data
        traj_data = self.dataset[idx]
        self.episode_data = traj_data
        self.episode_language_instruction = traj_data["instruction"]

        # Reset the AI2-THOR environment
        scene_name = traj_data["scene"]
        self._last_event = self.env.reset(scene=scene_name)

        # Set up the camera for multiview if enabled
        if self.multiview:
            event = self.env.step(action="GetMapViewCameraProperties", raise_for_failure=True)
            pose = event.metadata["actionReturn"].copy()
            pose["orthographic"] = True
            self.env.step(
                action="AddThirdPartyCamera",
                **pose,
                skyboxColor="white",
                raise_for_failure=True,
            )

        # Teleport the agent to the starting position
        pose = traj_data["agentPose"]
        self.env.step(
            action="Teleport",
            position={
                "x": pose["position"]["x"],
                "y": pose["position"]["y"],
                "z": pose["position"]["z"]
            },
            rotation={
                "x": 0,
                "y": pose["rotation"],
                "z": 0
            },
            horizon=pose["horizon"],
            standing=True
        )

        # Reset episode tracking information
        self._current_step = 0
        self.standing = True
        self.episode_log = []
        self._episode_start_time = time.time()
        self.img_paths = []
        self.total_reward = 0
        self.valid_actions = []
        self.reward = 0
        self._last_valid_action = None
        self._consecutive_action_count = 0
        self._episode_action_counts = {}
        
        return self._render(init_obs=True), {}

    def _compute_dense_reward(self, action_list, prev_distance, curr_distance, action_validity_error):
        dense_reward_mode = self.config.get("dense_reward_mode", "off")
        if dense_reward_mode not in {"anti_collapse_progress_v1", "anti_collapse_progress_v2"}:
            return {
                "dense_reward_total": 0.0,
                "dense_progress_reward": 0.0,
                "dense_repeat_action_penalty": 0.0,
                "dense_stagnation_repeat_penalty": 0.0,
                "dense_action_balance_penalty": 0.0,
                "dense_invalid_action_penalty": 0.0,
                "dense_distance_delta": 0.0,
            }

        progress_reward = 0.0
        distance_delta = prev_distance - curr_distance
        progress_epsilon = self.config.get("progress_epsilon", 0.01)
        if distance_delta > progress_epsilon:
            progress_reward = self.config.get("progress_reward", 0.02)
        elif distance_delta < -progress_epsilon:
            progress_reward = self.config.get("regress_penalty", -0.02)

        invalid_penalty = 0.0
        if action_validity_error != "ok" or not self.env.last_event.metadata.get("lastActionSuccess", False):
            invalid_penalty = self.config.get("invalid_action_penalty", -0.05)

        repeat_penalty = 0.0
        stagnation_repeat_penalty = 0.0
        action_balance_penalty = 0.0
        if action_list and action_validity_error == "ok":
            current_action = action_list[0].lower()
            if current_action == self._last_valid_action:
                self._consecutive_action_count += 1
            else:
                self._last_valid_action = current_action
                self._consecutive_action_count = 1

            self._episode_action_counts[current_action] = self._episode_action_counts.get(current_action, 0) + 1

            repeat_start = self.config.get("repeat_action_start", 3)
            if self._consecutive_action_count >= repeat_start:
                repeats_over = self._consecutive_action_count - repeat_start + 1
                repeat_penalty = self.config.get("repeat_action_penalty", -0.02) * repeats_over
                repeat_penalty = max(repeat_penalty, self.config.get("repeat_action_penalty_cap", -0.06))

            if dense_reward_mode == "anti_collapse_progress_v2":
                stagnation_start = self.config.get("stagnation_repeat_start", repeat_start)
                stagnation_eps = self.config.get("stagnation_delta_eps", 0.02)
                if (
                    self._consecutive_action_count >= stagnation_start
                    and distance_delta <= stagnation_eps
                ):
                    repeats_over = self._consecutive_action_count - stagnation_start + 1
                    stagnation_repeat_penalty = (
                        self.config.get("stagnation_repeat_penalty", -0.02) * repeats_over
                    )
                    stagnation_repeat_penalty = max(
                        stagnation_repeat_penalty,
                        self.config.get("stagnation_repeat_penalty_cap", -0.06),
                    )

                total_actions = sum(self._episode_action_counts.values())
                if total_actions >= self.config.get("action_balance_min_steps", 5):
                    top_share = max(self._episode_action_counts.values()) / total_actions
                    if top_share > self.config.get("action_top_share_penalty_threshold", 0.85):
                        action_balance_penalty += self.config.get("action_top_share_penalty", -0.01)

                    all_same_threshold = self.config.get("all_same_traj_penalty_threshold", 0.5)
                    if len(self._episode_action_counts) == 1 and top_share > all_same_threshold:
                        action_balance_penalty += self.config.get("all_same_traj_penalty", -0.02)

        dense_total = progress_reward + repeat_penalty + stagnation_repeat_penalty + action_balance_penalty + invalid_penalty
        return {
            "dense_reward_total": dense_total,
            "dense_progress_reward": progress_reward,
            "dense_repeat_action_penalty": repeat_penalty,
            "dense_stagnation_repeat_penalty": stagnation_repeat_penalty,
            "dense_action_balance_penalty": action_balance_penalty,
            "dense_invalid_action_penalty": invalid_penalty,
            "dense_distance_delta": distance_delta,
        }
    
    @env_state_reward_wrapper
    def step(self, action_str: str):
        """Execute an action in the environment.
        
        This method:
        1. Parses the raw LLM response to extract actions
        2. Executes each valid action in sequence
        3. Calculates rewards and metrics
        4. Generates the next observation
        
        Args:
            action_str: Raw text response from LLM
            
        Returns:
            Observation, reward, done, info
        """
        # Process the LLM response to extract actions
        rst = self.parse_func(
            response=action_str,
            special_token_list=self.config.get('special_token_list', None),
            action_sep=self.config.get('action_sep', ','),
            max_actions=self.config.get('max_actions_per_step', 1)
        )
        
        action_list = rst['actions']
        invalid_action_names = [
            action for action in action_list
            if action.lower() not in self.ACTION_LOOKUP
        ]
        if invalid_action_names:
            rst["format_correct"] = False
            rst["format_error_type"] = "invalid_action_name"
        prev_pos = self.env.last_event.metadata["agent"]["position"]
        _, prev_distance = self.measure_success()
        
        metrics = {
            "turn_metrics": {
                "action_is_valid": len(action_list) > 0,
                "action_is_effective": False,
                "format_correct": bool(rst.get("format_correct", False)),
                "too_many_actions": bool(rst.get("too_many_actions", False)),
                "action_count": len(action_list),
                "action_validity_error": (
                    "invalid_action_name"
                    if invalid_action_names
                    else ("ok" if action_list else "no_action")
                ),
            },
            "traj_metrics": {
                "success": False,
            }
        }
        
        self.reward = 0
        self.valid_actions = []
        done = False
        info = {}
        info.update(rst)
            
            
        # Execute valid actions
        if metrics["turn_metrics"]["action_is_valid"] and rst.get("format_correct", True):
            
            for action in action_list:
                action_lower = action.lower()
                if action_lower in self.ACTION_LOOKUP:
                    action_int = self.ACTION_LOOKUP[action_lower]
                    self._execute_action(action_int)
                    success, distance = self.measure_success()
                    
                    # Update reward based on success
                    if success:
                        self.reward += 10.0  # Success reward
                        done = True
                        metrics['traj_metrics']['success'] = True
                    
                    self.valid_actions.append(action)
                    
                    if done:
                        break
                else:
                    metrics['turn_metrics']['action_is_valid'] = False
                    metrics['turn_metrics']['action_validity_error'] = "invalid_action_name"
                    break
                
                self._current_step += 1
                if self._current_step >= self._max_episode_steps:
                    done = True
                    break
        
        if metrics['turn_metrics']['action_is_valid'] and rst.get("format_correct", True):
            self.reward += self.config.format_reward
            info["is_format_rewarded"] = True
        else:
            info["is_format_rewarded"] = False
            
        # Check if the agent position has changed (action was effective)
        curr_pos = self.env.last_event.metadata["agent"]["position"]
        metrics['turn_metrics']['action_is_effective'] = curr_pos["x"] != prev_pos["x"] or curr_pos["z"] != prev_pos["z"]
        
        # Update info dict
        info["metrics"] = metrics
        success, distance = self.measure_success()
        dense_metrics = self._compute_dense_reward(
            action_list=action_list,
            prev_distance=prev_distance,
            curr_distance=distance,
            action_validity_error=metrics['turn_metrics']['action_validity_error'],
        )
        if (
            self.config.get("dense_reward_mode", "off") == "off"
            and metrics['turn_metrics']['action_validity_error'] != "ok"
        ):
            dense_metrics["dense_invalid_action_penalty"] = self.config.get("invalid_action_penalty", -0.05)
            dense_metrics["dense_reward_total"] += dense_metrics["dense_invalid_action_penalty"]
        self.reward += dense_metrics["dense_reward_total"]
        metrics['turn_metrics'].update(dense_metrics)
        info['distance'] = distance
        info['instruction'] = self.episode_language_instruction
        info['env_step'] = self._current_step
        info['episode_elapsed_seconds'] = time.time() - self._episode_start_time
        info['task_success'] = success
        info['last_action_success'] = self.env.last_event.metadata['lastActionSuccess']
        info["env_feedback"] ="Last action is executed successfully." if info['last_action_success'] else "Last action is not executed successfully."
        self.info = info
        # Update total reward
        self.total_reward += self.reward
        
        return self._render(init_obs=False), self.reward, done, info
    
    def _execute_action(self, action_index):
        """Execute the discrete action in the environment.
        
        Args:
            action_index: Index of the action to execute
        """
        if action_index == 1:  # Move forward by 0.25 meter
            self._last_event = self.env.step(action="MoveAhead", moveMagnitude=self.step_length)
        elif action_index == 2:  # Move backward by 0.25 meter
            self._last_event = self.env.step(action="MoveBack", moveMagnitude=self.step_length)
        elif action_index == 3:  # Move right by 0.25 meter
            self._last_event = self.env.step(action="MoveRight", moveMagnitude=self.step_length)
        elif action_index == 4:  # Move left by 0.25 meter
            self._last_event = self.env.step(action="MoveLeft", moveMagnitude=self.step_length)
        elif action_index == 5:  # Rotate clockwise by 90 degrees
            self._last_event = self.env.step(action="RotateRight", degrees=90)
        elif action_index == 6:  # Rotate counterclockwise by 90 degrees
            self._last_event = self.env.step(action="RotateLeft", degrees=90)
        elif action_index == 7:  # Tilt the camera upward by 30 degrees
            self._last_event = self.env.step(action="LookUp", degrees=30)
        elif action_index == 8:  # Tilt the camera downward by 30 degrees
            self._last_event = self.env.step(action="LookDown", degrees=30)
    
    def measure_success(self):
        """Check if the agent has reached the target.
        
        Returns:
            success: Boolean indicating success
            distance: Distance to the target
        """
        agent_position = self.env.last_event.metadata["agent"]["position"]
        target_position = self.episode_data["target_position"]
        dist = math.sqrt(
            (agent_position["x"] - target_position["x"])**2 +
            (agent_position["z"] - target_position["z"])**2
        )
        success = (dist <= self.success_threshold)
        return float(success), dist
    
    def _render(self, init_obs=True):
        """Render the environment observation.
        
        This method creates either a text representation or an image of the environment
        state, depending on the configured render mode. It formats the observation string
        based on whether this is the initial observation or a subsequent one.
        
        Args:
            init_obs: Whether this is the initial observation
            
        Returns:
            Observation dict
        """
        img_placeholder = self.config.get("image_placeholder", "<image>")
        
        # Get format prompt without examples for action/init templates
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=False  # No examples for action and init obs
        )
        
        # Get the RGB frame from the environment
        frame = self.env.last_event.frame
        
        # Convert to PIL image for multimodal inputs
        multi_modal_data = {
            img_placeholder: [convert_numpy_to_PIL(frame)]
        }
        
        # Format the template
        if init_obs:
            obs_str = init_observation_template(
                observation=img_placeholder,
                instruction=self.episode_language_instruction,
                action_word="action" if self.config.max_actions_per_step == 1 else "action(s)",
            ) + "\n" + format_prompt_text
        else:
            obs_str = action_template(
                valid_action=self.valid_actions,
                observation=img_placeholder,
                reward=self.reward,
                done=self.measure_success()[0],
                instruction=self.episode_language_instruction,
                env_feedback=self.info["env_feedback"],
                action_word="action" if self.config.max_actions_per_step == 1 else "action(s)",
            ) + "\n" + format_prompt_text
        
        return {
            "obs_str": obs_str,
            "multi_modal_data": multi_modal_data
        }
    
    def system_prompt(self):
        """Get the system prompt for the environment.
        
        Returns a prompt explaining the environment to the LLM agent,
        with different prompts for text and vision modes.
        
        Returns:
            System prompt string
        """
        # Get format prompt with examples for system prompt
        format_prompt_text = self.format_prompt_func(
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
            add_example=True  # Always true for system prompt
        )
        
    
        return system_prompt(
            format=self.config.prompt_format,
            max_actions_per_step=self.config.max_actions_per_step,
            action_sep=self.config.action_sep,
        ) + '\n' + format_prompt_text
    
    def close(self):
        """Close the environment."""
        self.env.stop()
        
    def get_env_state(self):
        """
        Get the current state of the navigation environment focusing on visible objects.
        
        Returns:
            Dict: Contains target position, target direction, visible objects,
                and instruction information with rounded distances
        """
        # Get agent information
        agent_metadata = self.env.last_event.metadata["agent"]
        agent_position = agent_metadata["position"]
        agent_rotation = agent_metadata["rotation"]["y"]  # Only y-axis rotation is relevant
        
        # Get target information
        target_position = self.episode_data["target_position"]
        target_type = self.episode_data["targetObjectType"]
        success, distance = self.measure_success()
        
        # Calculate target's relative direction
        dx_target = target_position["x"] - agent_position["x"]
        dz_target = target_position["z"] - agent_position["z"]
        angle_to_target = math.degrees(math.atan2(dx_target, dz_target))
        relative_angle_target = (angle_to_target - agent_rotation) % 360
        if relative_angle_target > 180:
            relative_angle_target -= 360
            
        # Determine target's relative position
        if -45 <= relative_angle_target <= 45:
            target_relative_direction = "ahead"
        elif 45 < relative_angle_target <= 135:
            target_relative_direction = "right"
        elif -135 <= relative_angle_target < -45:
            target_relative_direction = "left"
        else:
            target_relative_direction = "back"
        
        # Get visible objects with position and relationship data
        objects = self.env.last_event.metadata["objects"]
        visible_objects = []
        
        for obj in objects:
            if obj.get("visible", False):
                obj_position = obj["position"]
                
                # Calculate distance from agent to object
                obj_distance = math.sqrt(
                    (agent_position["x"] - obj_position["x"])**2 +
                    (agent_position["z"] - obj_position["z"])**2
                )
                
                # Round distance to 2 decimal places
                obj_distance = round(obj_distance, 2)
                
                # Calculate relative angle to object (in degrees)
                dx = obj_position["x"] - agent_position["x"]
                dz = obj_position["z"] - agent_position["z"]
                angle_to_obj = math.degrees(math.atan2(dx, dz))
                # Adjust for agent's rotation (0 means directly in front)
                relative_angle = (angle_to_obj - agent_rotation) % 360
                if relative_angle > 180:
                    relative_angle -= 360
                    
                # Determine relative position (front, back, left, right)
                if -45 <= relative_angle <= 45:
                    relative_direction = "ahead"
                elif 45 < relative_angle <= 135:
                    relative_direction = "right"
                elif -135 <= relative_angle < -45:
                    relative_direction = "left"
                else:
                    relative_direction = "back"
                
                # Store object information
                visible_objects.append({
                    "type": obj["objectType"],
                    "direction_to_player": relative_direction,
                    "distance_to_player": obj_distance,
                })
    
        # Sort objects by distance (closest first)
        visible_objects.sort(key=lambda x: x["distance_to_player"])
        
        return {
            "target_obj_type": target_type, 
            "target_distance_to_player": round(distance, 2), 
            "target_direction_to_player": target_relative_direction,
            "visible_objects": visible_objects[:self.config.max_objects_in_state],   
        }


if __name__ == "__main__":
    # Example usage
    import os
    config = NavigationEnvConfig()
    env = NavigationEnv(config)
    print(env.system_prompt())
    
    obs, info = env.reset(seed=3)
    print(obs["obs_str"])
    i = 0
    os.makedirs("./test_navigation", exist_ok=True)
    img = obs["multi_modal_data"][config.image_placeholder][0]
    img.save(f"./test_navigation/navigation_{i}.png")
    done = False
    
    # Interactive testing loop
    while not done:
        i += 1
        action = input("Enter action (moveahead, moveback, moveright, moveleft, rotateright, rotateleft, lookup, lookdown): ")
        action = f"<think>Let me navigate toward the target.</think><answer>{action}</answer>"
        obs, reward, done, info = env.step(action)
        print(f"Reward: {reward}, Done: {done}")
        print(obs["obs_str"])
        img = obs["multi_modal_data"][config.image_placeholder][0]
        img.save(f"./test_navigation/navigation_{i}.png")
        print(f"Success: {info['metrics']['traj_metrics']['success']}, Action effective: {info['metrics']['turn_metrics']['action_is_effective']}")
        
        if done:
            break
    
    print(f"Total reward: {env.compute_reward()}")
    env.close()
