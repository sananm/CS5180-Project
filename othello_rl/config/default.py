import torch

BOARD_SIZE = 8
ACTION_SIZE = 65
CHANNELS = 3
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DEFAULT_SEED = 42
SHARED_CNN_CHANNELS = 128
SHARED_CNN_RES_BLOCKS = 5
SHARED_CNN_POLICY_HEAD_CHANNELS = 32
SHARED_CNN_VALUE_HIDDEN_DIM = 64
# AlphaZero hyperparameters (per CONTEXT.md decisions D-01 through D-09)
AZ_CPUCT = 1.0                    # PUCT exploration constant (D-01)
AZ_NUM_SIMS = 100                 # MCTS simulations per move (D-02: 100-200 range)
AZ_TEMP_THRESHOLD = 15            # Moves before temp->0 (D-03, scaled from Go's 30)
AZ_REPLAY_BUFFER_SIZE = 100_000   # FIFO buffer capacity (D-05)
AZ_BATCH_SIZE = 64                # Training batch size (D-09, start conservative)
AZ_LEARNING_RATE = 1e-3           # Adam LR (D-08)
AZ_EPOCHS_PER_ITER = 10           # Training epochs per iteration
AZ_GAMES_PER_ITER = 100           # Self-play games per iteration (D-06)

# PPO hyperparameters
PPO_CLIP_EPS = 0.2
PPO_GAMMA = 0.99
PPO_LAM = 0.95
PPO_EPOCHS = 4
PPO_MINIBATCH_SIZE = 64
PPO_LEARNING_RATE = 3e-4
PPO_VALUE_COEF = 0.5
PPO_ENTROPY_COEF = 0.01
PPO_MAX_GRAD_NORM = 0.5
PPO_EPISODES_PER_UPDATE = 10
PPO_CHECKPOINT_EVERY = 20

MINIMAX_DEPTHS = (4, 6)
TOURNAMENT_GAMES_PER_MATCHUP = 200
ARENA_MAX_MOVES = 128
ELO_INITIAL_RATING = 1500
ELO_K_FACTOR = 32
