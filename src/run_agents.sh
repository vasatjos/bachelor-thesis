#!/bin/bash

# ==========================================
# DQN
# ==========================================
DQN_BASE="--episodes 500_000 --log_each 1_000 --evaluate_for 20_000 --save_each 2_000 --seed 42 "

./run_thesis.sh dqn dqn1 $DQN_BASE  --epsilon 0.05 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh dqn dqn2 $DQN_BASE  --epsilon 0.05 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 4 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh dqn dqn3 $DQN_BASE  --epsilon 0.1 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh dqn dqn4 $DQN_BASE  --epsilon 0.1 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all --self_play --self_play_update_freq 2000 

# ==========================================
# DDQN
# ==========================================
DDQN_BASE="--episodes 500_000 --log_each 1_000 --evaluate_for 20_000 --save_each 2_000 --seed 42 "

./run_thesis.sh ddqn ddqn1 $DDQN_BASE  --epsilon 0.05 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 4 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh ddqn ddqn2 $DDQN_BASE  --epsilon 0.1 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh ddqn ddqn3 $DDQN_BASE  --epsilon 0.1 --gamma 0.99 --learning_rate 0.00005 --batch_size 32 --target_update_freq 100 --hidden_layer_count 4 --hidden_layer_size 1024 --hand_state_option full --played_subset all 

# ==========================================
# MONTE CARLO
# ==========================================
MC_BASE="--episodes 10_000_000_000 --log_each 50_000 --evaluate_for 1_000_000 --save_each 100_000 --seed 42 "

./run_thesis.sh monte_carlo mc1 $MC_BASE  --epsilon 0.01 --gamma 0.99 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc2 $MC_BASE  --epsilon 0.05 --gamma 0.99 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc3 $MC_BASE  --epsilon 0.1 --gamma 0.99 --every_visit --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc4 $MC_BASE  --epsilon 0.1 --gamma 0.99 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc5 $MC_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.01 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc6 $MC_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.05 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc7 $MC_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh monte_carlo mc8 $MC_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials --self_play --self_play_update_freq 10000 
./run_thesis.sh monte_carlo mc9 $MC_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials --self_play --self_play_update_freq 100000 

# ==========================================
# Q-LEARNING
# ==========================================
QL_BASE="--episodes 10_000_000_000 --log_each 50_000 --evaluate_for 1_000_000 --save_each 100_000 --seed 42 "

./run_thesis.sh q_learning ql1 $QL_BASE  --epsilon 0.01 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh q_learning ql2 $QL_BASE  --epsilon 0.05 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh q_learning ql3 $QL_BASE  --epsilon 0.1 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh q_learning ql4 $QL_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.01 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh q_learning ql5 $QL_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.05 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh q_learning ql6 $QL_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 
./run_thesis.sh q_learning ql7 $QL_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials --self_play --self_play_update_freq 10000 
./run_thesis.sh q_learning ql8 $QL_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.1 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials --self_play --self_play_update_freq 100000 
./run_thesis.sh q_learning ql9 $QL_BASE  --epsilon 0.5 --epsilon_decay 0.99999 --min_epsilon 0.001 --gamma 0.99 --alpha 0.2 --hand_state_option count_truncated --truncated_hand_size 4 --played_subset specials 

# ==========================================
# REINFORCE
# ==========================================
REINFORCE_BASE="--episodes 10_000_000 --log_each 500 --evaluate_for 10_000 --save_each 5000 --seed 42 "

./run_thesis.sh reinforce reinforce1 $REINFORCE_BASE  --gamma 0.99 --learning_rate 0.0001 --batch_size 32 --entropy_regularization 0.001 --baseline --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh reinforce reinforce2 $REINFORCE_BASE  --gamma 0.99 --learning_rate 0.0001 --batch_size 32 --entropy_regularization 0.01 --baseline --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh reinforce reinforce3 $REINFORCE_BASE  --gamma 0.99 --learning_rate 0.0001 --batch_size 32 --entropy_regularization 0.01 --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh reinforce reinforce4 $REINFORCE_BASE  --gamma 0.99 --learning_rate 0.0001 --batch_size 32 --entropy_regularization 0.05 --baseline --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 
./run_thesis.sh reinforce reinforce5 $REINFORCE_BASE  --gamma 0.99 --learning_rate 0.0001 --batch_size 32 --entropy_regularization 0.05 --baseline --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all --self_play --self_play_update_freq 2000 
./run_thesis.sh reinforce reinforce6 $REINFORCE_BASE  --gamma 0.99 --learning_rate 0.0001 --batch_size 32 --entropy_regularization 0.1 --baseline --hidden_layer_count 2 --hidden_layer_size 1024 --hand_state_option full --played_subset all 

echo "All training processes have been started in the background."
