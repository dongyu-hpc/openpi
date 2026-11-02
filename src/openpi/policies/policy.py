# from collections.abc import Sequence
# import logging
# import pathlib
# import time
# from typing import Any, TypeAlias

# import flax
# import flax.traverse_util
# import jax
# import jax.numpy as jnp
# import numpy as np
# from openpi_client import base_policy as _base_policy
# import torch
# from typing_extensions import override

# from openpi import transforms as _transforms
# from openpi.models import model as _model
# from openpi.shared import array_typing as at
# from openpi.shared import nnx_utils

# import inspect

# BasePolicy: TypeAlias = _base_policy.BasePolicy


# class Policy(BasePolicy):
#     def __init__(
#         self,
#         model: _model.BaseModel,
#         *,
#         rng: at.KeyArrayLike | None = None,
#         transforms: Sequence[_transforms.DataTransformFn] = (),
#         output_transforms: Sequence[_transforms.DataTransformFn] = (),
#         sample_kwargs: dict[str, Any] | None = None,
#         metadata: dict[str, Any] | None = None,
#         pytorch_device: str = "cpu",
#         is_pytorch: bool = False,
#     ):
#         """Initialize the Policy.

#         Args:
#             model: The model to use for action sampling.
#             rng: Random number generator key for JAX models. Ignored for PyTorch models.
#             transforms: Input data transformations to apply before inference.
#             output_transforms: Output data transformations to apply after inference.
#             sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
#             metadata: Additional metadata to store with the policy.
#             pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
#                           Only relevant when is_pytorch=True.
#             is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
#         """
#         self._model = model
#         self._input_transform = _transforms.compose(transforms)
#         self._output_transform = _transforms.compose(output_transforms)
#         self._sample_kwargs = sample_kwargs or {}
#         self._metadata = metadata or {}
#         self._is_pytorch_model = is_pytorch
#         self._pytorch_device = pytorch_device

#         if self._is_pytorch_model:
#             self._model = self._model.to(pytorch_device)
#             self._model.eval()
#             self._sample_actions = model.sample_actions
#         else:
#             # JAX model setup
#             self._sample_actions = nnx_utils.module_jit(model.sample_actions)
#             self._rng = rng or jax.random.key(0)

#     @override
#     def infer(self, obs: dict, *, noise: np.ndarray| None = None) -> dict:  # type: ignore[misc]
#         # Make a copy since transformations may modify the inputs in place.
#         inputs = jax.tree.map(lambda x: x, obs)
#         inputs = self._input_transform(inputs)
#         if not self._is_pytorch_model:
#             # Make a batch and convert to jax.Array.
#             inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
#             self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
#         else:
#             # Convert inputs to PyTorch tensors and move to correct device
#             inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
#             sample_rng_or_pytorch_device = self._pytorch_device

#         # Prepare kwargs for sample_actions
#         sample_kwargs = dict(self._sample_kwargs)

        
#         if noise is not None:
#             noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

#             if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
#                 noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
#             sample_kwargs["noise"] = noise

#         observation = _model.Observation.from_dict(inputs)
#         start_time = time.monotonic()
#         sample_result = self._sample_actions(
#             sample_rng_or_pytorch_device,
#             observation,
#             **sample_kwargs,
#         )
        
#         # Pi0 now returns (actions, velocity_history)
#         actions, velocity_history = sample_result
        
#         outputs = {
#             "state": inputs["state"],
#             "actions": actions,
#         }
        
        
#         model_time = time.monotonic() - start_time
#         if self._is_pytorch_model:
#             outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
#         else:
#             outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

#         print("[DEBUG] before output_transform:", outputs.keys())
#         outputs = self._output_transform(outputs)
#         print("[DEBUG] after output_transform:", outputs.keys())
        
#         outputs["policy_timing"] = {
#             "infer_ms": model_time * 1000,
#         }

#         outputs["velocity_history"] = np.asarray(velocity_history[0, ...])
#         return outputs


#     @property
#     def metadata(self) -> dict[str, Any]:
#         return self._metadata


# class PolicyRecorder(_base_policy.BasePolicy):
#     """Records the policy's behavior to disk."""

#     def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
#         self._policy = policy

#         logging.info(f"Dumping policy records to: {record_dir}")
#         self._record_dir = pathlib.Path(record_dir)
#         self._record_dir.mkdir(parents=True, exist_ok=True)
#         self._record_step = 0

#     @override
#     def infer(self, obs: dict) -> dict:  # type: ignore[misc]
#         results = self._policy.infer(obs)

#         data = {"inputs": obs, "outputs": results}
#         data = flax.traverse_util.flatten_dict(data, sep="/")

#         output_path = self._record_dir / f"step_{self._record_step}"
#         self._record_step += 1

#         np.save(output_path, np.asarray(data))
#         return results

# from collections.abc import Sequence
# import logging
# import pathlib
# import time
# from typing import Any, TypeAlias

# import flax
# import flax.traverse_util
# import jax
# import jax.numpy as jnp
# import numpy as np
# from openpi_client import base_policy as _base_policy
# import torch
# from typing_extensions import override

# from openpi import transforms as _transforms
# from openpi.models import model as _model
# from openpi.shared import array_typing as at
# from openpi.shared import nnx_utils

# BasePolicy: TypeAlias = _base_policy.BasePolicy


# class Policy(BasePolicy):
#     def __init__(
#         self,
#         model: _model.BaseModel,
#         *,
#         rng: at.KeyArrayLike | None = None,
#         transforms: Sequence[_transforms.DataTransformFn] = (),
#         output_transforms: Sequence[_transforms.DataTransformFn] = (),
#         sample_kwargs: dict[str, Any] | None = None,
#         metadata: dict[str, Any] | None = None,
#         pytorch_device: str = "cpu",
#         is_pytorch: bool = False,
#     ):
#         """Initialize the Policy.

#         Args:
#             model: The model to use for action sampling.
#             rng: Random number generator key for JAX models. Ignored for PyTorch models.
#             transforms: Input data transformations to apply before inference.
#             output_transforms: Output data transformations to apply after inference.
#             sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
#             metadata: Additional metadata to store with the policy.
#             pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
#                           Only relevant when is_pytorch=True.
#             is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
#         """
#         self._model = model
#         self._input_transform = _transforms.compose(transforms)
#         self._output_transform = _transforms.compose(output_transforms)
#         self._sample_kwargs = sample_kwargs or {}
#         self._metadata = metadata or {}
#         self._is_pytorch_model = is_pytorch
#         self._pytorch_device = pytorch_device

#         if self._is_pytorch_model:
#             self._model = self._model.to(pytorch_device)
#             self._model.eval()
#             self._sample_actions = model.sample_actions
#         else:
#             # JAX model setup
#             self._sample_actions = nnx_utils.module_jit(model.sample_actions)
#             self._rng = rng or jax.random.key(0)

#     @override
#     def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
#         # Make a copy since transformations may modify the inputs in place.
#         inputs = jax.tree.map(lambda x: x, obs)
#         inputs = self._input_transform(inputs)

#         if not self._is_pytorch_model:
#             # Make a batch and convert to jax.Array.
#             inputs = jax.tree.map(
#                 lambda x: jnp.asarray(x)[np.newaxis, ...], inputs
#             )
#             self._rng, sample_rng_or_pytorch_device = jax.random.split(
#                 self._rng
#             )
#         else:
#             # Convert inputs to PyTorch tensors and move to correct device.
#             inputs = jax.tree.map(
#                 lambda x: torch.from_numpy(np.array(x)).to(
#                     self._pytorch_device
#                 )[None, ...],
#                 inputs,
#             )
#             sample_rng_or_pytorch_device = self._pytorch_device

#         # Prepare kwargs for sample_actions.
#         sample_kwargs = dict(self._sample_kwargs)
        
#         if noise is not None:
#             noise = (
#                 torch.from_numpy(noise).to(self._pytorch_device)
#                 if self._is_pytorch_model
#                 else jnp.asarray(noise)
#             )

#             if noise.ndim == 2:
#                 # If noise is (action_horizon, action_dim), add batch dimension.
#                 noise = noise[None, ...]

#             sample_kwargs["noise"] = noise

#         observation = _model.Observation.from_dict(inputs)

#         # Time model sampling.
#         start_time = time.monotonic()
#         sample_actions = self._sample_actions(
#             sample_rng_or_pytorch_device,
#             observation,
#             **sample_kwargs,
#         )

#         # JAX execution can be asynchronous; wait for results before stopping.
#         if not self._is_pytorch_model:
#             jax.block_until_ready(sample_actions)

#         model_time = time.monotonic() - start_time

#         outputs = {
#             "state": inputs["state"],
#             "actions": sample_actions,
#         }

#         if self._is_pytorch_model:
#             outputs = jax.tree.map(
#                 lambda x: np.asarray(x[0, ...].detach().cpu()),
#                 outputs,
#             )
#         else:
#             outputs = jax.tree.map(
#                 lambda x: np.asarray(x[0, ...]),
#                 outputs,
#             )

#         outputs = self._output_transform(outputs)
#         # outputs["policy_timing"] = {
#         #     "infer_ms": model_time * 1000,
#         # }

#         actions = outputs["actions"]

#         outputs["policy_timing"] = {
#             "infer_ms": round(model_time * 1000, 3),
#             "generation_time_ms": round(model_time * 1000, 3),
#             "time_to_first_token_ms": None,
#             "inter_token_latency_ms": None,
#         }

#         outputs["policy_info"] = {
#             "generation_config": {
#                 "max_decoding_steps": self._sample_kwargs.get(
#                     "max_decoding_steps", 256
#                 ),
#                 "temperature": self._sample_kwargs.get(
#                     "temperature", 0.0
#                 ),
#             },
#             "generation_result": {
#                 "num_tokens_generated": None,
#                 "stop_reason": None,
#             },
#             "action_output": {
#                 "action_horizon": int(actions.shape[-2]),
#                 "action_dim": int(actions.shape[-1]),
#             },
#             "rtc": {
#                 "inference_timestamp": time.time(),
#                 "deadline_ms": None,
#                 "deadline_missed": None,
#             },
#         }

#         return outputs


#     @property
#     def metadata(self) -> dict[str, Any]:
#         return self._metadata


# class PolicyRecorder(_base_policy.BasePolicy):
#     """Records the policy's behavior to disk."""

#     def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
#         self._policy = policy

#         logging.info(f"Dumping policy records to: {record_dir}")
#         self._record_dir = pathlib.Path(record_dir)
#         self._record_dir.mkdir(parents=True, exist_ok=True)
#         self._record_step = 0

#     @override
#     def infer(self, obs: dict) -> dict:  # type: ignore[misc]
#         results = self._policy.infer(obs)

#         data = {"inputs": obs, "outputs": results}
#         data = flax.traverse_util.flatten_dict(data, sep="/")

#         output_path = self._record_dir / f"step_{self._record_step}"
#         self._record_step += 1

#         np.save(output_path, np.asarray(data))
#         return results


from collections.abc import Sequence
import inspect
import logging
import pathlib
import time
from typing import Any, TypeAlias

import flax
import flax.traverse_util
import jax
import jax.numpy as jnp
import numpy as np
from openpi_client import base_policy as _base_policy
import torch
from typing_extensions import override

from openpi import transforms as _transforms
from openpi.models import model as _model
from openpi.shared import array_typing as at
from openpi.shared import nnx_utils


BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
    """Policy wrapper supporting default and RTC generation modes."""

    def __init__(
        self,
        model: _model.BaseModel,
        *,
        rng: at.KeyArrayLike | None = None,
        transforms: Sequence[_transforms.DataTransformFn] = (),
        output_transforms: Sequence[_transforms.DataTransformFn] = (),
        sample_kwargs: dict[str, Any] | None = None,
        metadata: dict[str, Any] | None = None,
        pytorch_device: str = "cpu",
        is_pytorch: bool = False,
    ):
        """Initialize the policy.

        Args:
            model: Model used for action sampling.
            rng: Random key for JAX models; ignored for PyTorch.
            transforms: Input transformations.
            output_transforms: Output transformations.
            sample_kwargs: Sampling arguments. Supported policy-level options:
                generation_mode: "default" or "rtc".
                rtc_chunk_size: RTC chunk configuration, default 8.
                Other options are passed to model.sample_actions.
            metadata: Additional metadata.
            pytorch_device: PyTorch device.
            is_pytorch: Whether the model uses PyTorch.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)

        self._sample_kwargs = dict(sample_kwargs or {})
        self._generation_mode = self._sample_kwargs.pop(
            "generation_mode", "default"
        )
        self._rtc_chunk_size = int(
            self._sample_kwargs.pop("rtc_chunk_size", 8)
        )

        if self._generation_mode not in ("default", "rtc"):
            raise ValueError(
                "generation_mode must be 'default' or 'rtc'; "
                f"got {self._generation_mode!r}"
            )

        if self._rtc_chunk_size <= 0:
            raise ValueError(
                "rtc_chunk_size must be positive; "
                f"got {self._rtc_chunk_size}"
            )

        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions_default = self._model.sample_actions
            self._sample_actions_rtc = self._model.sample_actions
        else:
            # Keep the default inference entry point unchanged.
            self._sample_actions_default = nnx_utils.module_jit(
                self._model.sample_actions
            )

            # This entry point is only used for RTC calls. The RTC
            # implementation in the model must accept the corresponding
            # keyword arguments.
            self._sample_actions_rtc = nnx_utils.module_jit(
                self._model.sample_actions,
                static_argnames=("generation_mode", "rtc_chunk_size", "max_decoding_steps"),
            )

            self._rng = rng if rng is not None else jax.random.key(0)

        self._rtc_supported = self._supports_rtc_arguments(
            self._model.sample_actions
        )

        if self._generation_mode == "rtc" and not self._rtc_supported:
            raise TypeError(
                "RTC mode was requested, but model.sample_actions does not "
                "declare both 'generation_mode' and 'rtc_chunk_size'. "
                "Use a model with RTC support or select generation_mode='default'."
            )

    @staticmethod
    def _supports_rtc_arguments(sample_fn: Any) -> bool:
        """Check whether sample_actions explicitly accepts RTC arguments."""
        try:
            parameters = inspect.signature(sample_fn).parameters
        except (TypeError, ValueError):
            return False

        return (
            "generation_mode" in parameters
            and "rtc_chunk_size" in parameters
        )

    @override
    def infer(
        self,
        obs: dict,
        *,
        noise: np.ndarray | None = None,
    ) -> dict:  # type: ignore[misc]
        """Run inference and return actions plus diagnostic metadata."""
        # Copy inputs because transforms may modify them in place.
        inputs = jax.tree.map(lambda x: x, obs)
        inputs = self._input_transform(inputs)

        if not self._is_pytorch_model:
            # Add batch dimension and convert to JAX arrays.
            inputs = jax.tree.map(
                lambda x: jnp.asarray(x)[np.newaxis, ...],
                inputs,
            )
            self._rng, sample_rng_or_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to the selected device.
            inputs = jax.tree.map(
                lambda x: torch.from_numpy(np.array(x)).to(
                    self._pytorch_device
                )[None, ...],
                inputs,
            )
            sample_rng_or_device = self._pytorch_device

        sample_kwargs = dict(self._sample_kwargs)

        if noise is not None:
            noise_value = (
                torch.from_numpy(noise).to(self._pytorch_device)
                if self._is_pytorch_model
                else jnp.asarray(noise)
            )

            if noise_value.ndim == 2:
                noise_value = noise_value[None, ...]

            sample_kwargs["noise"] = noise_value

        observation = _model.Observation.from_dict(inputs)

        start_time = time.monotonic()

        if self._generation_mode == "rtc":
            sample_kwargs["generation_mode"] = "rtc"
            sample_kwargs["rtc_chunk_size"] = self._rtc_chunk_size

            sample_actions = self._sample_actions_rtc(
                sample_rng_or_device,
                observation,
                **sample_kwargs,
            )
        else:
            # Preserve the original default call path.
            sample_actions = self._sample_actions_default(
                sample_rng_or_device,
                observation,
                **sample_kwargs,
            )

        # JAX dispatch is asynchronous. Wait before measuring inference time.
        if not self._is_pytorch_model:
            jax.block_until_ready(sample_actions)

        model_time_ms = (time.monotonic() - start_time) * 1000.0

        outputs = {
            "state": inputs["state"],
            "actions": sample_actions,
        }

        if self._is_pytorch_model:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...].detach().cpu()),
                outputs,
            )
        else:
            outputs = jax.tree.map(
                lambda x: np.asarray(x[0, ...]),
                outputs,
            )

        outputs = self._output_transform(outputs)
        actions = outputs["actions"]

        outputs["policy_timing"] = {
            "infer_ms": round(model_time_ms, 3),
            "generation_time_ms": round(model_time_ms, 3),
            # These require measurements inside the generation loop.
            "time_to_first_token_ms": None,
            "inter_token_latency_ms": None,
        }

        outputs["policy_info"] = {
            "generation_config": {
                "generation_mode": self._generation_mode,
                "max_decoding_steps": self._sample_kwargs.get(
                    "max_decoding_steps", 256
                ),
                "temperature": self._sample_kwargs.get(
                    "temperature", 0.0
                ),
                "rtc_chunk_size": (
                    self._rtc_chunk_size
                    if self._generation_mode == "rtc"
                    else None
                ),
            },
            "generation_result": {
                # Populate these from actual model generation statistics
                # when the model exposes them.
                "num_tokens_generated": None,
                "stop_reason": None,
            },
            "action_output": {
                "action_horizon": int(actions.shape[-2]),
                "action_dim": int(actions.shape[-1]),
            },
            "rtc": {
                "enabled": self._generation_mode == "rtc",
                "inference_timestamp": time.time(),
                "deadline_ms": None,
                "deadline_missed": None,
            },
        }

        return outputs

    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records policy inputs and outputs to disk."""

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        record_dir: str,
    ):
        self._policy = policy

        logging.info("Dumping policy records to: %s", record_dir)
        self._record_dir = pathlib.Path(record_dir)
        self._record_dir.mkdir(parents=True, exist_ok=True)
        self._record_step = 0

    @override
    def infer(self, obs: dict) -> dict:  # type: ignore[misc]
        results = self._policy.infer(obs)

        data = {"inputs": obs, "outputs": results}
        data = flax.traverse_util.flatten_dict(data, sep="/")

        output_path = self._record_dir / f"step_{self._record_step}"
        self._record_step += 1

        np.save(output_path, np.asarray(data))
        return results