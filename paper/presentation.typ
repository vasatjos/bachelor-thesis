#import "@preview/diatypst:0.9.3": *

// #set text(font: "DejaVu Sans", lang: "en")

// #set text(size: 9pt)
#show: slides.with(
    title: "Reinforcement Learning For Prší Card Game",
    subtitle: "Josef Vašata",
    date: datetime.today().display("[day padding:none]. [month padding:none]. [year]"),
    authors: "Supervisor: Ing. Daniel Vašata Ph.D.",
    toc: false,
    theme: "full",
    count: "number"
)
#set heading(numbering: none)



// #set text(size: 11pt)


== Motivation
- Prší -- popular card game in the Czech Republic
- Interesting problem for AI
    - High level of stochasticity
    - Incomplete information

== Goals
+ Implement an environment for Prší
+ Compare multiple RL approaches:
    - Tabular methods
    - Deep learning methods
+ Evaluation:
    - Win-rate against greedy baseline
    - Win-rate against humans

== Formalizing The Problem
- The game is modeled as a _partially observeable Markov decision process_ (POMDP)

#align(center, image("images/pomdp.png", height: 50%))


#pagebreak()

- We define the return $G_t$ as the cummulative sum of rewards,
    the agent then tries to maximize $EE [G_t]$
/ *Return $G_t$*: $
        G_t = R_(t+1) + gamma R_(t+2) + gamma^2 R_(t+3) + ...
        = sum_(k=0)^infinity gamma^k R_(t+1+k)
    $
- For an agent's policy $pi$ we also define the action-value function $q_pi (s, a)$
/ *Action-value function $q_pi (s, a)$*: $
        q_pi (s, a) = EE_pi [G_t mid(bar) S_t = s, A_t = a]
    $

== Algorithms
- Tabular methods -- Monte Carlo, Q-Learning:
    - Based on estimating the action-value function through
        a learned estimate~Q
    - Descrete state space
- (Double) Deep Q-Network -- (D)DQN:
    - Learns to estimate Q
    - Uses neural networks
    - Similar states $=>$ similar Q estimate
- REINFORCE:
    - Policy gradient method (Directly optimize the policy
        $pi_bold(theta) (a mid(bar) s)$)

== Training Results
#grid(
    columns: (1.2fr, 1fr),
    gutter: 1em,
    [
        // TODO: concrete win-rates
        - *REINFORCE* achieved *65% win-rate* against the greedy agent
        - Tabular methods reached almost 50%
        - (D)DQN diverged (win-rate under 30%)
    ],
    [
        #image("images/reinforce_training.svg", width: 100%)
        #align(center)[#text(size: 9pt)[_REINFORCE win-rate throughout learning_]]
        #image("images/double_dqn_training.svg", width: 100%)
        #align(center)[#text(size: 9pt)[_DDQN win-rate throughout learning_]]
    ],
)

== Evaluation Against Human Players
- Performance comparison
    - Human vs. greedy agent: ~65% human win-rate
    - Human vs. REINFORCE: *~54%* human win-rate
- The REINFORCE agent lowered the human win-rate significantly,
    performance was almost even
// TODO: table

== Value of The Work
- Environment implementation:
    - Extensible Prší environment written in Python\ (Gymnasium-like API, self-play support)
- Evaluation against human players:
    - A custom CLI was created to compare agent performance to humans

#grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
        #image("images/cli-icons.png", width: 100%)
    ],
    [
        #image("images/cli-no-icons.png", width: 100%)
    ],
)
#align(center)[#text(size: 9pt)[_Human evaluation CLI with and without icons_]]


== Conclusion and Future work
    - Policy gradient methods managed to learn a non-trivial policy
      even through the game's high level of stochasticity
    - Future work:
        - State representation: Using RNNs
        - Advanced algorithms: Proximal Policy Optimization (PPO),
            Soft Actor-Critic (SAC), MuZero

= Thank you for your attention

== Opponent questions
- TODO
