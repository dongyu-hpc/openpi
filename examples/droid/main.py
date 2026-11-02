# # ruff: noqa

# import contextlib
# import dataclasses
# import datetime
# import faulthandler
# import os
# import signal
# import time
# from moviepy.editor import ImageSequenceClip
# import numpy as np
# from openpi_client import image_tools
# from openpi_client import websocket_client_policy
# import pandas as pd
# from PIL import Image
# from droid.robot_env import RobotEnv
# import tqdm
# import tyro

# faulthandler.enable()

# # DROID data collection frequency -- we slow down execution to match this frequency
# DROID_CONTROL_FREQUENCY = 15


# @dataclasses.dataclass
# class Args:
#     # Hardware parameters
#     left_camera_id: str = "<your_camera_id>"  # e.g., "24259877"
#     right_camera_id: str = "<your_camera_id>"  # e.g., "24514023"
#     wrist_camera_id: str = "<your_camera_id>"  # e.g., "13062452"

#     # Policy parameters
#     external_camera: str | None = (
#         None  # which external camera should be fed to the policy, choose from ["left", "right"]
#     )

#     # Rollout parameters
#     max_timesteps: int = 600
#     # How many actions to execute from a predicted action chunk before querying policy server again
#     # 8 is usually a good default (equals 0.5 seconds of action execution).
#     open_loop_horizon: int = 8

#     # Remote server parameters
#     remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
#     remote_port: int = (
#         8000  # point this to the port of the policy server, default server port for openpi servers is 8000
#     )


# # We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# # waiting for a new action chunk, it will raise an exception and the server connection dies.
# # This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
# @contextlib.contextmanager
# def prevent_keyboard_interrupt():
#     """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
#     interrupted = False
#     original_handler = signal.getsignal(signal.SIGINT)

#     def handler(signum, frame):
#         nonlocal interrupted
#         interrupted = True

#     signal.signal(signal.SIGINT, handler)
#     try:
#         yield
#     finally:
#         signal.signal(signal.SIGINT, original_handler)
#         if interrupted:
#             raise KeyboardInterrupt


# def main(args: Args):
#     # Make sure external camera is specified by user -- we only use one external camera for the policy
#     assert (
#         args.external_camera is not None and args.external_camera in ["left", "right"]
#     ), f"Please specify an external camera to use for the policy, choose from ['left', 'right'], but got {args.external_camera}"

#     # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
#     env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")
#     print("Created the droid env!")

#     # Connect to the policy server
#     policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

#     df = pd.DataFrame(columns=["success", "duration", "video_filename"])

#     while True:
#         instruction = input("Enter instruction: ")

#         # Rollout parameters
#         actions_from_chunk_completed = 0
#         pred_action_chunk = None

#         # Prepare to save video of rollout
#         timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
#         video = []
#         bar = tqdm.tqdm(range(args.max_timesteps))
#         print("Running rollout... press Ctrl+C to stop early.")
#         for t_step in bar:
#             start_time = time.time()
#             try:
#                 # Get the current observation
#                 curr_obs = _extract_observation(
#                     args,
#                     env.get_observation(),
#                     # Save the first observation to disk
#                     save_to_disk=t_step == 0,
#                 )

#                 video.append(curr_obs[f"{args.external_camera}_image"])

#                 # Send websocket request to policy server if it's time to predict a new chunk
#                 if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= args.open_loop_horizon:
#                     actions_from_chunk_completed = 0

#                     # We resize images on the robot laptop to minimize the amount of data sent to the policy server
#                     # and improve latency.
#                     request_data = {
#                         "observation/exterior_image_1_left": image_tools.resize_with_pad(
#                             curr_obs[f"{args.external_camera}_image"], 224, 224
#                         ),
#                         "observation/wrist_image_left": image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224),
#                         "observation/joint_position": curr_obs["joint_position"],
#                         "observation/gripper_position": curr_obs["gripper_position"],
#                         "prompt": instruction,
#                     }

#                     # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
#                     # Ctrl+C will be handled after the server call is complete
#                     with prevent_keyboard_interrupt():
#                         # this returns action chunk [10, 8] of 10 joint velocity actions (7) + gripper position (1)
#                         pred_action_chunk = policy_client.infer(request_data)["actions"]
#                     assert pred_action_chunk.shape == (10, 8)

#                 # Select current action to execute from chunk
#                 action = pred_action_chunk[actions_from_chunk_completed]
#                 actions_from_chunk_completed += 1

#                 # Binarize gripper action
#                 if action[-1].item() > 0.5:
#                     # action[-1] = 1.0
#                     action = np.concatenate([action[:-1], np.ones((1,))])
#                 else:
#                     # action[-1] = 0.0
#                     action = np.concatenate([action[:-1], np.zeros((1,))])

#                 # clip all dimensions of action to [-1, 1]
#                 action = np.clip(action, -1, 1)

#                 env.step(action)

#                 # Sleep to match DROID data collection frequency
#                 elapsed_time = time.time() - start_time
#                 if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
#                     time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
#             except KeyboardInterrupt:
#                 break

#         video = np.stack(video)
#         save_filename = "video_" + timestamp
#         ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

#         success: str | float | None = None
#         while not isinstance(success, float):
#             success = input(
#                 "Did the rollout succeed? (enter y for 100%, n for 0%), or a numeric value 0-100 based on the evaluation spec"
#             )
#             if success == "y":
#                 success = 1.0
#             elif success == "n":
#                 success = 0.0

#             success = float(success) / 100
#             if not (0 <= success <= 1):
#                 print(f"Success must be a number in [0, 100] but got: {success * 100}")

#         df = df.append(
#             {
#                 "success": success,
#                 "duration": t_step,
#                 "video_filename": save_filename,
#             },
#             ignore_index=True,
#         )

#         if input("Do one more eval? (enter y or n) ").lower() != "y":
#             break
#         env.reset()

#     os.makedirs("results", exist_ok=True)
#     timestamp = datetime.datetime.now().strftime("%I:%M%p_%B_%d_%Y")
#     csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
#     df.to_csv(csv_filename)
#     print(f"Results saved to {csv_filename}")


# def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
#     image_observations = obs_dict["image"]
#     left_image, right_image, wrist_image = None, None, None
#     for key in image_observations:
#         # Note the "left" below refers to the left camera in the stereo pair.
#         # The model is only trained on left stereo cams, so we only feed those.
#         if args.left_camera_id in key and "left" in key:
#             left_image = image_observations[key]
#         elif args.right_camera_id in key and "left" in key:
#             right_image = image_observations[key]
#         elif args.wrist_camera_id in key and "left" in key:
#             wrist_image = image_observations[key]

#     # Drop the alpha dimension
#     left_image = left_image[..., :3]
#     right_image = right_image[..., :3]
#     wrist_image = wrist_image[..., :3]

#     # Convert to RGB
#     left_image = left_image[..., ::-1]
#     right_image = right_image[..., ::-1]
#     wrist_image = wrist_image[..., ::-1]

#     # In addition to image observations, also capture the proprioceptive state
#     robot_state = obs_dict["robot_state"]
#     cartesian_position = np.array(robot_state["cartesian_position"])
#     joint_position = np.array(robot_state["joint_positions"])
#     gripper_position = np.array([robot_state["gripper_position"]])

#     # Save the images to disk so that they can be viewed live while the robot is running
#     # Create one combined image to make live viewing easy
#     if save_to_disk:
#         combined_image = np.concatenate([left_image, wrist_image, right_image], axis=1)
#         combined_image = Image.fromarray(combined_image)
#         combined_image.save("robot_camera_views.png")

#     return {
#         "left_image": left_image,
#         "right_image": right_image,
#         "wrist_image": wrist_image,
#         "cartesian_position": cartesian_position,
#         "joint_position": joint_position,
#         "gripper_position": gripper_position,
#     }

# if __name__ == "__main__":
#     args: Args = tyro.cli(Args)
#     main(args)


# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import os
import signal
import time
from concurrent.futures import ThreadPoolExecutor, Future

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
import pandas as pd
from PIL import Image
from droid.robot_env import RobotEnv
import tqdm
import tyro

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15
ACTION_CHUNK_LENGTH = 10


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "<your_camera_id>"
    right_camera_id: str = "<your_camera_id>"
    wrist_camera_id: str = "<your_camera_id>"

    # Policy parameters
    external_camera: str | None = None

    # Rollout parameters
    max_timesteps: int = 600
    open_loop_horizon: int = 8

    # Remote server parameters
    remote_host: str = "0.0.0.0"
    remote_port: int = 8000

    # RTC diagnostics
    log_timing: bool = True


@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Delay Ctrl+C until the protected call completes."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


def build_request(args: Args, curr_obs, instruction: str):
    """Build a policy request from one captured observation."""
    return {
        "observation/exterior_image_1_left": image_tools.resize_with_pad(
            curr_obs[f"{args.external_camera}_image"], 224, 224
        ),
        "observation/wrist_image_left": image_tools.resize_with_pad(
            curr_obs["wrist_image"], 224, 224
        ),
        "observation/joint_position": curr_obs["joint_position"],
        "observation/gripper_position": curr_obs["gripper_position"],
        "prompt": instruction,
    }


def infer_timed(policy_client, request_data):
    """Run one synchronous inference and return timing metadata."""
    start = time.perf_counter()
    result = policy_client.infer(request_data)
    elapsed = time.perf_counter() - start

    actions = np.asarray(result["actions"])
    if actions.shape != (ACTION_CHUNK_LENGTH, 8):
        raise ValueError(
            f"Expected actions shape {(ACTION_CHUNK_LENGTH, 8)}, "
            f"got {actions.shape}"
        )

    return {
        "actions": actions,
        "inference_seconds": elapsed,
        "completed_at": time.perf_counter(),
    }


def execute_action(env, action):
    """Apply the same action preprocessing as the original script."""
    action = np.asarray(action).copy()

    # Binarize gripper action.
    if action[-1] > 0.5:
        action[-1] = 1.0
    else:
        action[-1] = 0.0

    # Clip all dimensions to [-1, 1].
    action = np.clip(action, -1, 1)
    env.step(action)


def main(args: Args):
    assert (
        args.external_camera is not None
        and args.external_camera in ["left", "right"]
    ), (
        "Please specify external_camera as 'left' or 'right'; "
        f"got {args.external_camera}"
    )

    if not 1 <= args.open_loop_horizon <= ACTION_CHUNK_LENGTH:
        raise ValueError(
            f"open_loop_horizon must be between 1 and {ACTION_CHUNK_LENGTH}"
        )

    env = RobotEnv(
        action_space="joint_velocity",
        gripper_action_space="position",
    )
    print("Created the droid env!")

    policy_client = websocket_client_policy.WebsocketClientPolicy(
        args.remote_host,
        args.remote_port,
    )

    # One background inference at a time. This avoids concurrent requests
    # against a single websocket policy client.
    executor = ThreadPoolExecutor(max_workers=1)

    results = []

    try:
        while True:
            instruction = input("Enter instruction: ")

            actions_from_chunk_completed = 0
            pred_action_chunk = None
            pending_future: Future | None = None

            inference_times = []
            deadline_misses = 0
            action_step_times = []
            chunk_switch_jumps = []

            timestamp = datetime.datetime.now().strftime(
                "%Y_%m_%d_%H-%M-%S"
            )
            video = []

            bar = tqdm.tqdm(range(args.max_timesteps))
            print("Running rollout... press Ctrl+C to stop early.")

            try:
                for t_step in bar:
                    loop_start = time.perf_counter()

                    # Capture the current observation.
                    curr_obs = _extract_observation(
                        args,
                        env.get_observation(),
                        save_to_disk=(t_step == 0),
                    )
                    video.append(curr_obs[f"{args.external_camera}_image"])

                    # Request the first chunk synchronously.
                    if pred_action_chunk is None:
                        request_data = build_request(
                            args, curr_obs, instruction
                        )

                        with prevent_keyboard_interrupt():
                            result = infer_timed(
                                policy_client, request_data
                            )

                        pred_action_chunk = result["actions"]
                        inference_times.append(
                            result["inference_seconds"]
                        )
                        actions_from_chunk_completed = 0

                    # When the current chunk reaches the configured
                    # replanning horizon, switch to the prefetched chunk.
                    if (
                        actions_from_chunk_completed
                        >= args.open_loop_horizon
                    ):
                        if pending_future is None:
                            # This should be unusual; submit from the latest
                            # observation rather than reusing stale input.
                            request_data = build_request(
                                args, curr_obs, instruction
                            )
                            pending_future = executor.submit(
                                infer_timed,
                                policy_client,
                                request_data,
                            )

                        if not pending_future.done():
                            deadline_misses += 1
                            if args.log_timing:
                                print(
                                    f"[deadline miss] t={t_step}: "
                                    "next chunk not ready; waiting"
                                )

                        # Wait for the new chunk. We deliberately do not
                        # repeat the previous action while waiting.
                        with prevent_keyboard_interrupt():
                            result = pending_future.result()

                        pending_future = None

                        new_chunk = result["actions"]
                        inference_times.append(
                            result["inference_seconds"]
                        )

                        # Diagnostic only: compare the last action of the
                        # old chunk with the first action of the new chunk.
                        # This is not a smoothing operation.
                        old_last = pred_action_chunk[
                            actions_from_chunk_completed - 1, :7
                        ]
                        new_first = new_chunk[0, :7]
                        jump = float(
                            np.max(np.abs(new_first - old_last))
                        )
                        chunk_switch_jumps.append(jump)

                        pred_action_chunk = new_chunk
                        actions_from_chunk_completed = 0

                    # Execute one action from the active chunk.
                    action = pred_action_chunk[
                        actions_from_chunk_completed
                    ]
                    actions_from_chunk_completed += 1

                    execute_action(env, action)

                    action_elapsed = time.perf_counter() - loop_start
                    action_step_times.append(action_elapsed)

                    # Prefetch the next chunk while the current chunk is
                    # being executed. Do not submit multiple concurrent jobs.
                    if (
                        pending_future is None
                        and actions_from_chunk_completed
                        >= args.open_loop_horizon
                    ):
                        # The current observation is the latest available
                        # observation at this point in the loop.
                        request_data = build_request(
                            args, curr_obs, instruction
                        )
                        pending_future = executor.submit(
                            infer_timed,
                            policy_client,
                            request_data,
                        )

                    # Keep the nominal control cadence.
                    elapsed = time.perf_counter() - loop_start
                    period = 1.0 / DROID_CONTROL_FREQUENCY
                    if elapsed < period:
                        time.sleep(period - elapsed)

                    if args.log_timing:
                        bar.set_postfix(
                            infer_ms=(
                                round(inference_times[-1] * 1000, 1)
                                if inference_times
                                else 0
                            ),
                            misses=deadline_misses,
                        )

            except KeyboardInterrupt:
                print("Rollout interrupted by user.")

            finally:
                # Do not leave a future running against the websocket
                # after this rollout ends.
                if pending_future is not None:
                    try:
                        pending_future.result()
                    except Exception as exc:
                        print(f"Background inference ended with: {exc}")

            # Save video.
            if video:
                video_array = np.stack(video)
                save_filename = f"video_{timestamp}"
                ImageSequenceClip(
                    list(video_array), fps=10
                ).write_videofile(
                    save_filename + ".mp4",
                    codec="libx264",
                )
            else:
                save_filename = f"video_{timestamp}"

            # Summarize timing and switching diagnostics.
            mean_infer_ms = (
                float(np.mean(inference_times) * 1000)
                if inference_times
                else float("nan")
            )
            p95_infer_ms = (
                float(np.percentile(inference_times, 95) * 1000)
                if inference_times
                else float("nan")
            )
            mean_step_ms = (
                float(np.mean(action_step_times) * 1000)
                if action_step_times
                else float("nan")
            )
            max_switch_jump = (
                float(np.max(chunk_switch_jumps))
                if chunk_switch_jumps
                else float("nan")
            )

            print("\n--- RTC timing summary ---")
            print(f"Inference calls: {len(inference_times)}")
            print(f"Mean inference: {mean_infer_ms:.1f} ms")
            print(f"P95 inference: {p95_infer_ms:.1f} ms")
            print(f"Mean action-loop time: {mean_step_ms:.1f} ms")
            print(f"Deadline misses: {deadline_misses}")
            print(f"Max chunk-boundary jump (first 7 dims): {max_switch_jump}")

            # Record an evaluation result.
            success_input = input(
                "Did the rollout succeed? "
                "(y=100%, n=0%, or enter a number from 0 to 100): "
            ).strip().lower()

            if success_input == "y":
                success = 1.0
            elif success_input == "n":
                success = 0.0
            else:
                success = float(success_input) / 100.0

            if not 0 <= success <= 1:
                raise ValueError(
                    f"Success must be in [0, 100], got {success * 100}"
                )

            results.append(
                {
                    "success": success,
                    "duration": t_step + 1,
                    "video_filename": save_filename,
                    "mean_inference_ms": mean_infer_ms,
                    "p95_inference_ms": p95_infer_ms,
                    "mean_action_loop_ms": mean_step_ms,
                    "deadline_misses": deadline_misses,
                    "max_chunk_boundary_jump": max_switch_jump,
                }
            )

            if input("Do one more eval? (enter y or n) ").lower() != "y":
                break

            env.reset()

    finally:
        executor.shutdown(wait=True)

    os.makedirs("results", exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%I-%M%p_%B_%d_%Y")
    csv_filename = os.path.join("results", f"eval_{timestamp}.csv")
    pd.DataFrame(results).to_csv(csv_filename, index=False)
    print(f"Results saved to {csv_filename}")


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    left_image, right_image, wrist_image = None, None, None

    for key in image_observations:
        if args.left_camera_id in key and "left" in key:
            left_image = image_observations[key]
        elif args.right_camera_id in key and "left" in key:
            right_image = image_observations[key]
        elif args.wrist_camera_id in key and "left" in key:
            wrist_image = image_observations[key]

    if left_image is None or right_image is None or wrist_image is None:
        raise RuntimeError(
            "Could not find all required camera images. Check camera IDs."
        )

    # Drop alpha and convert BGR to RGB.
    left_image = left_image[..., :3][..., ::-1]
    right_image = right_image[..., :3][..., ::-1]
    wrist_image = wrist_image[..., :3][..., ::-1]

    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    joint_position = np.array(robot_state["joint_positions"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if save_to_disk:
        combined_image = np.concatenate(
            [left_image, wrist_image, right_image], axis=1
        )
        Image.fromarray(combined_image).save("robot_camera_views.png")

    return {
        "left_image": left_image,
        "right_image": right_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "joint_position": joint_position,
        "gripper_position": gripper_position,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    main(args)