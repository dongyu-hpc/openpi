from collections.abc import Sequence
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

import inspect

BasePolicy: TypeAlias = _base_policy.BasePolicy


class Policy(BasePolicy):
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
        """Initialize the Policy.

        Args:
            model: The model to use for action sampling.
            rng: Random number generator key for JAX models. Ignored for PyTorch models.
            transforms: Input data transformations to apply before inference.
            output_transforms: Output data transformations to apply after inference.
            sample_kwargs: Additional keyword arguments to pass to model.sample_actions.
            metadata: Additional metadata to store with the policy.
            pytorch_device: Device to use for PyTorch models (e.g., "cpu", "cuda:0").
                          Only relevant when is_pytorch=True.
            is_pytorch: Whether the model is a PyTorch model. If False, assumes JAX model.
        """
        self._model = model
        self._input_transform = _transforms.compose(transforms)
        self._output_transform = _transforms.compose(output_transforms)
        self._sample_kwargs = sample_kwargs or {}
        self._metadata = metadata or {}
        self._is_pytorch_model = is_pytorch
        self._pytorch_device = pytorch_device

        if self._is_pytorch_model:
            self._model = self._model.to(pytorch_device)
            self._model.eval()
            self._sample_actions = model.sample_actions
        else:
            # JAX model setup
            self._sample_actions = nnx_utils.module_jit(model.sample_actions)
            self._rng = rng or jax.random.key(0)

    # @override
    # def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
    #     # Make a copy since transformations may modify the inputs in place.
    #     inputs = jax.tree.map(lambda x: x, obs)
    #     inputs = self._input_transform(inputs)
    #     if not self._is_pytorch_model:
    #         # Make a batch and convert to jax.Array.
    #         inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
    #         self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
    #     else:
    #         # Convert inputs to PyTorch tensors and move to correct device
    #         inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
    #         sample_rng_or_pytorch_device = self._pytorch_device

    #     # Prepare kwargs for sample_actions
    #     sample_kwargs = dict(self._sample_kwargs)
    #     if noise is not None:
    #         noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

    #         if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
    #             noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
    #         sample_kwargs["noise"] = noise

    #     observation = _model.Observation.from_dict(inputs)
    #     start_time = time.monotonic()
    #     outputs = {
    #         "state": inputs["state"],
    #         "actions": self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs),
    #     }
    #     model_time = time.monotonic() - start_time
    #     if self._is_pytorch_model:
    #         outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
    #     else:
    #         outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

    #     outputs = self._output_transform(outputs)
    #     outputs["policy_timing"] = {
    #         "infer_ms": model_time * 1000,
    #     }
    #     return outputs

    @override
    def infer(self, obs: dict, *, noise: np.ndarray | None = None) -> dict:  # type: ignore[misc]
        # Make a copy since transformations may modify the inputs in place.
        inputs = jax.tree.map(lambda x: x, obs)

        # RTC 参数不是观测,先取出,避免进入 input transform。
        rtc_prev = inputs.pop("rtc_prev_chunk", None)
        rtc_d = inputs.pop("rtc_d", None)
        rtc_leftover = inputs.pop("rtc_leftover", None)

        inputs = self._input_transform(inputs)
        if not self._is_pytorch_model:
            # Make a batch and convert to jax.Array.
            inputs = jax.tree.map(lambda x: jnp.asarray(x)[np.newaxis, ...], inputs)
            self._rng, sample_rng_or_pytorch_device = jax.random.split(self._rng)
        else:
            # Convert inputs to PyTorch tensors and move to correct device
            inputs = jax.tree.map(lambda x: torch.from_numpy(np.array(x)).to(self._pytorch_device)[None, ...], inputs)
            sample_rng_or_pytorch_device = self._pytorch_device

        # Prepare kwargs for sample_actions
        sample_kwargs = dict(self._sample_kwargs)
        if noise is not None:
            noise = torch.from_numpy(noise).to(self._pytorch_device) if self._is_pytorch_model else jnp.asarray(noise)

            if noise.ndim == 2:  # If noise is (action_horizon, action_dim), add batch dimension
                noise = noise[None, ...]  # Make it (1, action_horizon, action_dim)
            sample_kwargs["noise"] = noise

        # RTC: 把上一个块的剩余部分、延迟 d、有效步数传给 sample_actions。
        if rtc_prev is not None:
            if rtc_d is None or rtc_leftover is None:
                raise ValueError("rtc_prev_chunk 需要同时提供 rtc_d 和 rtc_leftover")
            if self._is_pytorch_model:
                raise NotImplementedError("RTC 补丁只写了 JAX 版")
            if "prev_chunk" not in inspect.signature(self._model.sample_actions).parameters:
                raise NotImplementedError(
                    f"{type(self._model).__name__}.sample_actions 不支持 RTC。"
                    "pi0-FAST 是自回归解码,请改用 pi0 或 pi05 的配置。"
                )

            # msgpack 传过来的标量可能是 0 维数组,统一转成 int
            d, leftover = int(np.asarray(rtc_d)), int(np.asarray(rtc_leftover))
            horizon, dim = self._model.action_horizon, self._model.action_dim
            prev = np.asarray(rtc_prev, dtype=np.float32)
            if prev.shape != (horizon, dim):
                raise ValueError(f"rtc_prev_chunk 形状应为 {(horizon, dim)},实际为 {prev.shape}")
            if not 0 <= d <= leftover <= horizon:
                raise ValueError(f"需要 0 <= d <= leftover <= {horizon},实际 d={d}, leftover={leftover}")

            sample_kwargs["prev_chunk"] = jnp.asarray(prev)[None, ...]
            sample_kwargs["rtc_d"] = jnp.asarray(d, dtype=jnp.int32)
            sample_kwargs["rtc_leftover"] = jnp.asarray(leftover, dtype=jnp.int32)

        observation = _model.Observation.from_dict(inputs)
        start_time = time.monotonic()
        actions = self._sample_actions(sample_rng_or_pytorch_device, observation, **sample_kwargs)
        if not self._is_pytorch_model:
            # JAX 是异步派发,不等待的话 infer_ms 只统计了派发时间
            actions = jax.block_until_ready(actions)
        model_time = time.monotonic() - start_time

        outputs = {
            "state": inputs["state"],
            "actions": actions,
        }
        if self._is_pytorch_model:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...].detach().cpu()), outputs)
        else:
            outputs = jax.tree.map(lambda x: np.asarray(x[0, ...]), outputs)

        # output transform 之前的动作(归一化空间,未裁剪维度),RTC 的引导必须用它。
        # output transform 可能原地修改,所以先拷贝。
        model_actions = np.array(outputs["actions"], dtype=np.float32, copy=True)

        outputs = self._output_transform(outputs)
        outputs["model_actions"] = model_actions
        outputs["policy_timing"] = {
            "infer_ms": model_time * 1000,
        }
        return outputs


    @property
    def metadata(self) -> dict[str, Any]:
        return self._metadata


class PolicyRecorder(_base_policy.BasePolicy):
    """Records the policy's behavior to disk."""

    def __init__(self, policy: _base_policy.BasePolicy, record_dir: str):
        self._policy = policy

        logging.info(f"Dumping policy records to: {record_dir}")
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
