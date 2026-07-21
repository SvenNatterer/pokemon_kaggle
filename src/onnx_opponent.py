import os
import gc
import numpy as np

try:
    import onnxruntime as ort
    HAS_ONNXRUNTIME = True
except ImportError:
    HAS_ONNXRUNTIME = False

import torch


class ONNXOpponentWrapper:
    """Fast GIL-releasing ONNX Runtime wrapper for opponent models."""

    def __init__(self, onnx_path: str, is_recurrent: bool = True, lstm_hidden_size: int = 256, torch_model=None):
        self.onnx_path = onnx_path
        self.is_recurrent = is_recurrent
        self.lstm_hidden_size = lstm_hidden_size
        self.session = None
        self.torch_model = torch_model
        self.observation_space = getattr(torch_model, "observation_space", None) if torch_model is not None else None
        policy_obj = getattr(torch_model, "policy", None) if torch_model is not None else None
        self.structured_options = bool(getattr(policy_obj, "structured_options", False))

        if HAS_ONNXRUNTIME and os.path.exists(onnx_path):
            sess_options = ort.SessionOptions()
            sess_options.intra_op_num_threads = 1
            sess_options.inter_op_num_threads = 1
            self.session = ort.InferenceSession(onnx_path, sess_options, providers=["CPUExecutionProvider"])
            input_nodes = {node.name: node for node in self.session.get_inputs()}
            input_names = set(input_nodes.keys())
            self.is_structured = "entity_ids" in input_names or "hand_ids" in input_names
            action_dim = 66
            if "action_mask" in input_nodes:
                shape = input_nodes["action_mask"].shape
                if len(shape) >= 2 and isinstance(shape[1], int):
                    action_dim = shape[1]
            from gym import spaces as gym_spaces
            self.action_space = gym_spaces.Discrete(action_dim)
            # Once the ONNX session is created, release the heavy PyTorch model to save RAM
            self.torch_model = None
            gc.collect()
        else:
            self.is_structured = False
            self.action_space = None

    @property
    def policy(self):
        class DummyPolicy:
            def __init__(self, structured_options):
                self.structured_options = structured_options
        return DummyPolicy(self.structured_options)

    @classmethod
    def get_or_export(cls, model_path: str, torch_model):
        """Load an existing ONNX model or export from PyTorch if possible."""
        if model_path is None or torch_model is None:
            return torch_model
        from src.agents.rule_based_agent import is_rule_based_model_spec
        if is_rule_based_model_spec(model_path):
            return torch_model
        onnx_path = model_path.replace(".zip", ".onnx")

        policy = getattr(torch_model, "policy", torch_model)
        is_recurrent = hasattr(policy, "lstm_actor")
        lstm_hidden_size = policy.lstm_actor.hidden_size if is_recurrent else 256

        if not os.path.exists(onnx_path) and torch_model is not None:
            cls.export_sb3_policy_to_onnx(torch_model, onnx_path)

        if os.path.exists(onnx_path) and HAS_ONNXRUNTIME:
            return cls(onnx_path, is_recurrent=is_recurrent, lstm_hidden_size=lstm_hidden_size, torch_model=torch_model)
        return torch_model

    @staticmethod
    def export_sb3_policy_to_onnx(torch_model, onnx_path: str):
        """Export SB3 policy network to ONNX format."""
        try:
            policy = getattr(torch_model, "policy", torch_model)
            policy.eval()

            obs_space = getattr(torch_model, "observation_space", None)
            if obs_space is None:
                return

            is_recurrent = hasattr(policy, "lstm_actor")
            lstm_hidden_size = policy.lstm_actor.hidden_size if is_recurrent else 256

            if hasattr(obs_space, "spaces"):
                keys = list(obs_space.spaces.keys())
                dummy_inputs = []
                input_names = []
                dynamic_axes = {}

                for key in keys:
                    space = obs_space.spaces[key]
                    is_int = (
                        space.dtype == np.int8
                        or space.dtype == np.int32
                        or space.dtype == np.int64
                    )
                    tensor = torch.zeros(
                        (1, *space.shape),
                        dtype=torch.int64 if is_int else torch.float32,
                    )
                    dummy_inputs.append(tensor)
                    input_names.append(key)
                    dynamic_axes[key] = {0: "batch_size"}

                if is_recurrent:
                    from sb3_contrib.common.recurrent.type_aliases import RNNStates

                    h_pi = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    c_pi = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    h_vf = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    c_vf = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    starts = torch.ones((1, 1), dtype=torch.float32)

                    for name, tensor, shape in [
                        ("h_pi", h_pi, {1: "batch_size"}),
                        ("c_pi", c_pi, {1: "batch_size"}),
                        ("h_vf", h_vf, {1: "batch_size"}),
                        ("c_vf", c_vf, {1: "batch_size"}),
                        ("starts", starts, {0: "batch_size"}),
                    ]:
                        dummy_inputs.append(tensor)
                        input_names.append(name)
                        dynamic_axes[name] = shape

                    output_names = [
                        "action",
                        "new_h_pi",
                        "new_c_pi",
                        "new_h_vf",
                        "new_c_vf",
                    ]
                    for name in output_names:
                        dynamic_axes[name] = {0: "batch_size"}

                    class RecurrentPolicyWrapper(torch.nn.Module):
                        def __init__(self, pol, keys):
                            super().__init__()
                            self.pol = pol
                            self.keys = keys

                        def forward(self, *args):
                            num_keys = len(self.keys)
                            obs_dict = {key: args[i] for i, key in enumerate(self.keys)}
                            h_pi, c_pi, h_vf, c_vf, starts = args[num_keys:]
                            rnn_states = RNNStates(pi=(h_pi, c_pi), vf=(h_vf, c_vf))
                            actions, values, log_probs, new_states = self.pol(
                                obs_dict, rnn_states, starts
                            )
                            return (
                                actions,
                                new_states.pi[0],
                                new_states.pi[1],
                                new_states.vf[0],
                                new_states.vf[1],
                            )

                    wrapper = RecurrentPolicyWrapper(policy, keys)
                else:
                    output_names = ["action"]
                    dynamic_axes["action"] = {0: "batch_size"}

                    class FeedForwardPolicyWrapper(torch.nn.Module):
                        def __init__(self, pol, keys):
                            super().__init__()
                            self.pol = pol
                            self.keys = keys

                        def forward(self, *args):
                            obs_dict = {key: args[i] for i, key in enumerate(self.keys)}
                            actions, _, _ = self.pol(obs_dict)
                            return actions

                    wrapper = FeedForwardPolicyWrapper(policy, keys)

                torch.onnx.export(
                    wrapper,
                    tuple(dummy_inputs),
                    onnx_path,
                    opset_version=14,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    dynamo=False,
                )
            else:
                dummy_input = torch.zeros((1, *obs_space.shape), dtype=torch.float32)
                input_names = ["obs"]
                dynamic_axes = {"obs": {0: "batch_size"}}

                if is_recurrent:
                    from sb3_contrib.common.recurrent.type_aliases import RNNStates

                    h_pi = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    c_pi = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    h_vf = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    c_vf = torch.zeros((1, 1, lstm_hidden_size), dtype=torch.float32)
                    starts = torch.ones((1, 1), dtype=torch.float32)

                    dummy_inputs = (dummy_input, h_pi, c_pi, h_vf, c_vf, starts)
                    input_names.extend(["h_pi", "c_pi", "h_vf", "c_vf", "starts"])
                    for name in ["h_pi", "c_pi", "h_vf", "c_vf"]:
                        dynamic_axes[name] = {1: "batch_size"}
                    dynamic_axes["starts"] = {0: "batch_size"}

                    output_names = ["action", "new_h_pi", "new_c_pi", "new_h_vf", "new_c_vf"]
                    for name in output_names:
                        dynamic_axes[name] = {0: "batch_size"}

                    class SingleBoxRecurrentPolicyWrapper(torch.nn.Module):
                        def __init__(self, pol):
                            super().__init__()
                            self.pol = pol

                        def forward(self, obs, h_pi, c_pi, h_vf, c_vf, starts):
                            rnn_states = RNNStates(pi=(h_pi, c_pi), vf=(h_vf, c_vf))
                            actions, values, log_probs, new_states = self.pol(obs, rnn_states, starts)
                            return actions, new_states.pi[0], new_states.pi[1], new_states.vf[0], new_states.vf[1]

                    wrapper = SingleBoxRecurrentPolicyWrapper(policy)
                else:
                    dummy_inputs = (dummy_input,)
                    output_names = ["action"]
                    dynamic_axes["action"] = {0: "batch_size"}

                    class SingleBoxFeedForwardPolicyWrapper(torch.nn.Module):
                        def __init__(self, pol):
                            super().__init__()
                            self.pol = pol

                        def forward(self, obs):
                            actions, _, _ = self.pol(obs)
                            return actions

                    wrapper = SingleBoxFeedForwardPolicyWrapper(policy)

                torch.onnx.export(
                    wrapper,
                    dummy_inputs,
                    onnx_path,
                    opset_version=14,
                    input_names=input_names,
                    output_names=output_names,
                    dynamic_axes=dynamic_axes,
                    dynamo=False,
                )

        except Exception as e:
            print(f"Failed to export ONNX model {onnx_path}: {e}")
            if os.path.exists(onnx_path):
                try:
                    os.remove(onnx_path)
                except OSError:
                    pass

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        if self.session is None:
            if self.torch_model is None:
                raise RuntimeError("ONNXOpponentWrapper has no session and no fallback PyTorch model")
            return self.torch_model.predict(
                observation, state=state, episode_start=episode_start, deterministic=deterministic
            )

        try:
            inputs = {}
            input_specs = {inp.name: inp.type for inp in self.session.get_inputs()}

            for name, type_str in input_specs.items():
                if name in observation:
                    arr = observation[name]
                    if "int" in type_str:
                        inputs[name] = arr.astype(np.int64)
                    else:
                        inputs[name] = arr.astype(np.float32)
                elif "tensor" in observation and "tensor" in input_specs:
                    inputs["tensor"] = observation["tensor"].astype(np.float32)
                elif name == "h_pi":
                    inputs[name] = (
                        state[0]
                        if state is not None
                        else np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
                    )
                elif name == "c_pi":
                    inputs[name] = (
                        state[1]
                        if state is not None
                        else np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
                    )
                elif name == "h_vf":
                    inputs[name] = (
                        state[2]
                        if state is not None
                        else np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
                    )
                elif name == "c_vf":
                    inputs[name] = (
                        state[3]
                        if state is not None
                        else np.zeros((1, 1, self.lstm_hidden_size), dtype=np.float32)
                    )
                elif name == "starts":
                    starts_val = (
                        episode_start
                        if episode_start is not None
                        else np.ones((1, 1), dtype=np.float32)
                    )
                    if starts_val.ndim == 1:
                        starts_val = np.expand_dims(starts_val, axis=-1)
                    inputs[name] = starts_val.astype(np.float32)

            outputs = self.session.run(None, inputs)

            if self.is_recurrent:
                action = outputs[0]
                new_state = (outputs[1], outputs[2], outputs[3], outputs[4])
                return action, new_state
            else:
                action = outputs[0]
                return action, state
        except Exception as e:
            if self.torch_model is not None:
                return self.torch_model.predict(
                    observation, state=state, episode_start=episode_start, deterministic=deterministic
                )
            raise RuntimeError(f"ONNX inference failed: {e}")
