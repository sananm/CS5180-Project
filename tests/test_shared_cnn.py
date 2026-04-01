import torch

from othello_rl.alphazero.network import build_alphazero_network
from othello_rl.config import ACTION_SIZE, BOARD_SIZE, CHANNELS
from othello_rl.game.othello_game import OthelloGame
from othello_rl.models.shared_cnn import SharedCNN
from othello_rl.ppo.network import build_ppo_network
from othello_rl.utils.tensor_utils import board_to_tensor


def test_shared_cnn_forward_shapes():
    model = SharedCNN()
    x = torch.randn(2, CHANNELS, BOARD_SIZE, BOARD_SIZE)

    policy, value = model(x)

    assert policy.shape == (2, ACTION_SIZE)
    assert value.shape == (2, 1)


def test_shared_cnn_value_is_bounded():
    model = SharedCNN()
    x = torch.randn(4, CHANNELS, BOARD_SIZE, BOARD_SIZE)

    _, value = model(x)

    assert torch.all(value <= 1.0)
    assert torch.all(value >= -1.0)


def test_shared_cnn_uses_action_size_65():
    model = SharedCNN()
    x = torch.randn(1, CHANNELS, BOARD_SIZE, BOARD_SIZE)

    policy, _ = model(x)

    assert model.action_size == ACTION_SIZE == 65
    assert policy.shape[1] == 65


def test_shared_cnn_accepts_board_tensor_from_tensor_utils():
    game = OthelloGame(8)
    board = game.getCanonicalForm(game.getInitBoard(), 1)
    tensor = board_to_tensor(board).unsqueeze(0)
    model = SharedCNN()

    policy, value = model(tensor)

    assert policy.shape == (1, ACTION_SIZE)
    assert value.shape == (1, 1)


def test_shared_cnn_backward_pass_populates_gradients():
    torch.manual_seed(0)
    model = SharedCNN()
    x = torch.randn(4, CHANNELS, BOARD_SIZE, BOARD_SIZE)
    target_actions = torch.randint(0, ACTION_SIZE, (4,))
    target_values = torch.rand(4, 1) * 2 - 1

    policy, value = model(x)
    loss = torch.nn.CrossEntropyLoss()(policy, target_actions)
    loss = loss + torch.nn.MSELoss()(value, target_values)
    loss.backward()

    assert any(parameter.grad is not None for parameter in model.conv_in.parameters())
    assert any(parameter.grad is not None for parameter in model.res_blocks.parameters())
    assert any(parameter.grad is not None for parameter in model.policy_head.parameters())
    assert any(parameter.grad is not None for parameter in model.value_head.parameters())


def test_alphazero_builder_returns_sharedcnn():
    model = build_alphazero_network()
    assert isinstance(model, SharedCNN)


def test_ppo_builder_returns_sharedcnn():
    model = build_ppo_network()
    assert isinstance(model, SharedCNN)


def test_alphazero_and_ppo_networks_have_identical_parameter_shapes():
    alphazero_model = build_alphazero_network()
    ppo_model = build_ppo_network()

    alphazero_state = alphazero_model.state_dict()
    ppo_state = ppo_model.state_dict()

    assert list(alphazero_state.keys()) == list(ppo_state.keys())
    assert {
        name: tuple(tensor.shape) for name, tensor in alphazero_state.items()
    } == {
        name: tuple(tensor.shape) for name, tensor in ppo_state.items()
    }
