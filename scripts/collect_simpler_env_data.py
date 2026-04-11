import requests
import os
import argparse
import json
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import imageio
import sys

# Because we import get_args, we will actually let it parse first and patch its parser if possible
# Alternatively, we just extract out our arguments manually before calling get_args
_sys_argv_copy = sys.argv.copy()

custom_parser = argparse.ArgumentParser(add_help=False)
custom_parser.add_argument("--target-successes", type=int, default=100)
custom_parser.add_argument("--output-dir", type=str, default="./data/simpler_drawer_successes")
custom_parser.add_argument("--port", type=int, default=10093)

custom_args, remaining_argv = custom_parser.parse_known_args()

sys.argv = [sys.argv[0]] + remaining_argv

from simpler_env.evaluation.argparse import get_args
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict


def save_lerobot_data(
    output_dir, episode_id, images, raw_actions, task_description, fps=3
):
    # Setup directory structure
    meta_dir = os.path.join(output_dir, "meta")
    data_chunk_dir = os.path.join(output_dir, "data", "000")
    video_dir = os.path.join(data_chunk_dir, "video.image_0")

    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(data_chunk_dir, exist_ok=True)
    os.makedirs(video_dir, exist_ok=True)

    # Save video
    episode_str = f"episode_{episode_id:06d}"
    video_path = os.path.join(video_dir, f"{episode_str}.mp4")
    imageio.mimwrite(video_path, images, fps=fps, codec="libx264")

    # Save parquet
    timestamps = [i / float(fps) for i in range(len(raw_actions))]

    parquet_data = {
        "timestamp": timestamps,
        "annotation.human.action.task_description": [task_description] * len(raw_actions),
    }

    # Extract actions from raw_actions list of dicts
    action_x = [a["world_vector"][0] for a in raw_actions]
    action_y = [a["world_vector"][1] for a in raw_actions]
    action_z = [a["world_vector"][2] for a in raw_actions]

    # Check if we have rotation_delta
    if isinstance(raw_actions[0], dict) and "rotation_delta" in raw_actions[0]:
        action_roll = [a["rotation_delta"][0] for a in raw_actions]
        action_pitch = [a["rotation_delta"][1] for a in raw_actions]
        action_yaw = [a["rotation_delta"][2] for a in raw_actions]
    else:
        # Fallback to zero
        action_roll = [0.0] * len(raw_actions)
        action_pitch = [0.0] * len(raw_actions)
        action_yaw = [0.0] * len(raw_actions)

    if isinstance(raw_actions[0], dict) and "gripper_closedness_action" in raw_actions[0]:
        action_gripper = [a["gripper_closedness_action"][0] for a in raw_actions]
    elif isinstance(raw_actions[0], dict) and "open_gripper" in raw_actions[0]:
        action_gripper = [a["open_gripper"][0] for a in raw_actions]
    elif isinstance(raw_actions[0], dict) and "gripper" in raw_actions[0]:
        action_gripper = [a["gripper"][0] for a in raw_actions]
    else:
        action_gripper = [0.0] * len(raw_actions)

    parquet_data["action.x"] = action_x
    parquet_data["action.y"] = action_y
    parquet_data["action.z"] = action_z
    parquet_data["action.roll"] = action_roll
    parquet_data["action.pitch"] = action_pitch
    parquet_data["action.yaw"] = action_yaw
    parquet_data["action.gripper"] = action_gripper

    table = pa.Table.from_pydict(parquet_data)
    parquet_path = os.path.join(data_chunk_dir, f"{episode_str}.parquet")
    pq.write_table(table, parquet_path)

    # Update meta files
    episodes_file = os.path.join(meta_dir, "episodes.jsonl")
    with open(episodes_file, "a") as f:
        f.write(json.dumps({"episode_id": episode_id, "length": len(raw_actions)}) + "\n")

    tasks_file = os.path.join(meta_dir, "tasks.jsonl")
    with open(tasks_file, "a") as f:
        f.write(json.dumps({"task_id": episode_id, "task": task_description}) + "\n")

    # Overwrite modality and info just to ensure they exist
    modality_file = os.path.join(meta_dir, "modality.json")
    if not os.path.exists(modality_file):
        with open(modality_file, "w") as f:
            json.dump({
                "video_keys": ["image_0"],
                "action_keys": ["x", "y", "z", "roll", "pitch", "yaw", "gripper"],
                "state_keys": [],
                "language_keys": ["task_description"]
            }, f, indent=4)

    stats_file = os.path.join(meta_dir, "stats_gr00t.json")
    if not os.path.exists(stats_file):
        with open(stats_file, "w") as f:
            json.dump({}, f)  # Dummy stats

    info_file = os.path.join(meta_dir, "info.json")
    if not os.path.exists(info_file):
        with open(info_file, "w") as f:
            json.dump({
                "fps": fps,
                "total_episodes": 0,  # Could update this at the end
            }, f, indent=4)


def run_collection(args):
    # Setup model
    os.environ["DISPLAY"] = ""
    if args.policy_model == "rt1" and not args.port:
        import tensorflow as tf
        os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
        gpus = tf.config.list_physical_devices("GPU")
        if len(gpus) > 0:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=args.tf_memory_limit)],
            )
        from simpler_env.policies.rt1.rt1_model import RT1Inference
        model = RT1Inference(
            saved_model_path=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )
    elif args.port:
        # Fallback to server-based inference if port is provided, regardless of policy model, since user may omit --policy-model server
        from simpler_env.policies.octo.octo_server_model import OctoServerInference
        model = OctoServerInference(
            model_type=args.ckpt_path,
            policy_setup=args.policy_setup,
            action_scale=args.action_scale,
        )

        def _query_for_action(image_primary, text, goal=None, modality="l"):
            fake_pay_load = model._get_fake_pay_load(image_primary, text, modality)
            url = f"http://localhost:{args.port}/query"
            reply = requests.post(url, json=fake_pay_load, timeout=100).json()
            from simpler_env.policies.octo.octo_server_model import loads
            return loads(reply)

        def _reset(task_description):
            model.task = task_description
            model.sticky_action_is_on = False
            model.gripper_action_repeat = 0
            model.sticky_gripper_action = 0.0
            model.previous_gripper_action = None
            url = f"http://localhost:{args.port}/reset"
            try:
                requests.post(url, timeout=100)
            except Exception:
                pass
        model._query_for_action = _query_for_action
        model.reset = _reset

    elif "octo" in args.policy_model:
        if args.ckpt_path is None or args.ckpt_path == "None":
            args.ckpt_path = args.policy_model
        if "server" in args.policy_model:
            from simpler_env.policies.octo.octo_server_model import OctoServerInference
            model = OctoServerInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                action_scale=args.action_scale,
            )
        else:
            from simpler_env.policies.octo.octo_model import OctoInference
            model = OctoInference(
                model_type=args.ckpt_path,
                policy_setup=args.policy_setup,
                init_rng=args.octo_init_rng,
                action_scale=args.action_scale,
            )
    elif "openvla" in args.policy_model:
        from simpler_env.policies.openvla.openvla_model import OPENVLAInference
        model = OPENVLAInference(policy_setup=args.policy_setup)
    else:
        raise NotImplementedError()

    control_mode = get_robot_control_mode(args.robot, args.policy_model)

    # Base kwargs for environment
    additional_env_build_kwargs = args.additional_env_build_kwargs or {}
    env_kwargs = dict(
        obs_mode="rgbd",
        robot=args.robot,
        sim_freq=args.sim_freq,
        control_mode=control_mode,
        control_freq=args.control_freq,
        max_episode_steps=args.max_episode_steps,
        scene_name=args.scene_name,
        camera_cfgs={"add_segmentation": True},
        rgb_overlay_path=args.rgb_overlay_path,
    )
    if args.enable_raytracing:
        ray_tracing_dict = {"shader_dir": "rt"}
        ray_tracing_dict.update(additional_env_build_kwargs)
        additional_env_build_kwargs = ray_tracing_dict

    env = build_maniskill2_env(
        args.env_name,
        **additional_env_build_kwargs,
        **env_kwargs,
    )

    successes_collected = 0
    target_successes = getattr(args, "target_successes", 100)
    output_dir = getattr(args, "output_dir", "./data/simpler_drawer_successes")

    print(f"Starting data collection. Target successes: {target_successes}")

    episode_id_counter = 0

    # Ensure output directory is clean if it exists? We leave it up to user or just append.
    # We will append, but to be clean, let's make sure it exists.
    os.makedirs(output_dir, exist_ok=True)

    for obj_episode_id in range(args.obj_episode_range[0], args.obj_episode_range[1]):
        if successes_collected >= target_successes:
            break

        for robot_init_x in args.robot_init_xs:
            for robot_init_y in args.robot_init_ys:
                for robot_init_quat in args.robot_init_quats:
                    if successes_collected >= target_successes:
                        break

                    env_reset_options = {
                        "robot_init_options": {
                            "init_xy": np.array([robot_init_x, robot_init_y]),
                            "init_rot_quat": robot_init_quat,
                        },
                        "obj_init_options": {
                            "episode_id": obj_episode_id,
                        }
                    }

                    obs, _ = env.reset(options=env_reset_options)
                    is_final_subtask = env.is_final_subtask()

                    task_description = env.get_language_instruction()
                    model.reset(task_description)

                    image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
                    images = [image]
                    raw_actions = []

                    predicted_terminated, done, truncated = False, False, False
                    timestep = 0

                    while not (predicted_terminated or truncated):
                        raw_action, action = model.step(image, task_description)
                        raw_actions.append(raw_action)

                        predicted_terminated = bool(action["terminate_episode"][0] > 0)
                        if predicted_terminated:
                            if not is_final_subtask:
                                predicted_terminated = False
                                env.advance_to_next_subtask()

                        obs, reward, done, truncated, info = env.step(
                            np.concatenate([action["world_vector"], action["rot_axangle"], action["gripper"]]),
                        )

                        new_task_description = env.get_language_instruction()
                        if new_task_description != task_description:
                            task_description = new_task_description
                        is_final_subtask = env.is_final_subtask()

                        image = get_image_from_maniskill2_obs_dict(env, obs, camera_name=args.obs_camera_name)
                        images.append(image)
                        timestep += 1

                        if done or truncated:
                            break

                    if info.get("success", False):
                        # Success
                        print(f"Episode {obj_episode_id} successful! Saving data...")
                        save_lerobot_data(
                            output_dir,
                            episode_id_counter,
                            images[:-1],
                            raw_actions,
                            task_description,
                            fps=args.control_freq
                        )
                        episode_id_counter += 1
                        successes_collected += 1
                        print(f"Successes collected: {successes_collected} / {target_successes}")
                    else:
                        print(f"Episode {obj_episode_id} failed.")

    # Update total episodes in info.json at the end
    meta_dir = os.path.join(output_dir, "meta")
    info_file = os.path.join(meta_dir, "info.json")
    if os.path.exists(info_file):
        with open(info_file, "r") as f:
            info = json.load(f)
        info["total_episodes"] = successes_collected
        with open(info_file, "w") as f:
            json.dump(info, f, indent=4)


if __name__ == "__main__":
    args = get_args()
    args.target_successes = custom_args.target_successes
    args.output_dir = custom_args.output_dir
    args.port = custom_args.port

    run_collection(args)
