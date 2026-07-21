import numpy as np
import torch

from src.custom_policy import PokemonTCGFeatureExtractor
from src.custom_ppo import CustomPPO, PokemonTCGRecurrentPolicy
from src.env_wrapper import V6_ACTION_SPACE_SIZE, PokemonTCGEnv


def scalar_env():
    return PokemonTCGEnv(
        [6] * 60,
        [5] * 60,
        action_space_size=V6_ACTION_SPACE_SIZE,
        structured_v2=False,
    )


def zero_observation(env):
    observation = {
        key: np.zeros(space.shape, dtype=space.dtype)
        for key, space in env.observation_space.spaces.items()
    }
    observation["action_mask"][0] = 1
    return observation


def tensors(observation):
    return {
        key: torch.as_tensor(np.asarray(value)).unsqueeze(0)
        for key, value in observation.items()
    }


def evolution_pair(extractor):
    names = extractor.card_name_tokens.cpu().numpy()
    evolves_from = extractor.card_evolves_from_tokens.cpu().numpy()
    name_to_card = {
        int(token): card_id
        for card_id, token in enumerate(names)
        if int(token) > 0
    }
    for evolved_card, token in enumerate(evolves_from):
        if int(token) in name_to_card:
            return name_to_card[int(token)], evolved_card
    raise AssertionError("card catalogue contains no resolvable evolution pair")


def test_legacy_scalar_extractor_keeps_original_layout():
    env = scalar_env()
    extractor = PokemonTCGFeatureExtractor(env.observation_space)

    assert not extractor.structured_v2
    assert not extractor.scalar_embeddings
    assert not extractor.supports_option_embeddings
    assert extractor.net[0].in_features == 1500


def test_scalar_embeddings_produce_finite_features_and_gradients():
    env = scalar_env()
    observation = zero_observation(env)
    observation["vector"][300] = 6
    observation["vector"][306] = 6
    observation["vector"][801] = 6
    extractor = PokemonTCGFeatureExtractor(
        env.observation_space, scalar_embeddings=True
    )

    features = extractor(tensors(observation))
    assert features.shape == (1, 256)
    assert torch.isfinite(features).all()
    features.square().mean().backward()
    assert extractor.scalar_card_embedding.weight.grad is not None
    assert torch.count_nonzero(extractor.scalar_card_embedding.weight.grad) > 0


def test_scalar_embeddings_keep_fast_scalar_main_path():
    env = scalar_env()
    extractor = PokemonTCGFeatureExtractor(
        env.observation_space, scalar_embeddings=True
    )

    assert extractor.net[0].in_features == 1500
    assert extractor.net[2].out_features == 256
    assert extractor.scalar_field_card_indices[0].item() == 300
    assert extractor.scalar_hand_card_indices[0].item() == 306
    assert extractor.scalar_option_bases[0].item() + 1 == 801
    assert extractor.scalar_option_bases[0].item() + 6 == 806


def test_scalar_relations_detect_identity_evolution_and_attack_ownership():
    env = scalar_env()
    extractor = PokemonTCGFeatureExtractor(
        env.observation_space, scalar_embeddings=True
    )
    base_card, evolved_card = evolution_pair(extractor)
    attack_ids = extractor.card_attack_ids[evolved_card]
    attack_id = int(attack_ids[attack_ids > 0][0]) if (attack_ids > 0).any() else 0

    observation = zero_observation(env)
    observation["vector"][300] = base_card
    observation["vector"][306] = evolved_card
    observation["vector"][801] = evolved_card
    observation["vector"][806] = attack_id
    relation_features = extractor._scalar_option_relations(
        tensors(observation)["vector"].float(),
        torch.tensor([[evolved_card] + [0] * 64]),
        torch.tensor([[attack_id] + [0] * 64]),
    )

    assert relation_features[0, 0, 0].item() > 0  # same card in hand
    assert relation_features[0, 0, 3].item() == 1  # evolves from field card
    if attack_id:
        assert relation_features[0, 0, 4].item() == 1  # attack belongs to card


def test_scalar_embeddings_enable_shared_v6_option_scorer():
    env = scalar_env()
    model = CustomPPO(
        PokemonTCGRecurrentPolicy,
        env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        device="cpu",
        policy_kwargs={
            "features_extractor_class": PokemonTCGFeatureExtractor,
            "features_extractor_kwargs": {"scalar_embeddings": True},
        },
    )

    assert model.policy.structured_options
    assert model.policy.lightweight_options
    assert model.policy.option_state_projection.out_features == 32
    assert model.policy.option_bias.out_features == 1
