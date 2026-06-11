#import "@preview/diatypst:0.9.3": *

#show: slides.with(
    title: "Reinforcement Learning For Prší Card Game",
    subtitle: "Josef Vašata",
    date: datetime.today().display("[day padding:none]. [month padding:none]. [year]"),
    authors: "Supervisor: Ing. Daniel Vašata Ph.D.",
    toc: false,
    theme: "full",
    count: none,
)
#set heading(numbering: none)


== Motivation
- Prší -- popular card game in the Czech Republic
- Interesting problem for AI
    - High level of stochasticity
    - Incomplete information

#v(2.5em)
#align(center, image("images/kings.jpg", height: 40%))
#v(-0.8em)
#align(center)[#text(size: 9pt)[_King cards of each suit in a Prší deck_]]

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

- We define the return $G_t$ as the cummulative sum of (discounted) rewards,
    $G_t = sum_(k=0)^infinity gamma^k R_(t+1+k)$
    , the agent then tries to maximize $EE [G_t]$

- For an agent's policy $pi$ we also define the action-value function
    \ $q_pi (s, a) = EE_pi [G_t mid(bar) S_t = s, A_t = a]$


== Algorithms
- Tabular methods -- Monte Carlo, Q-Learning:
    - Based on estimating the action-value function through
        a learned Q table
    - Descrete state space
- (Double) Deep Q-Network:
    - Learns to estimate Q as a function
    - Uses neural networks
    - Similar states $=>$ similar Q estimate
- REINFORCE:
    - Policy gradient method -- directly optimizes the policy
        $pi_bold(theta) (a mid(bar) s)$


== Training Results
#grid(
    columns: (1.21fr, 1fr),
    gutter: 1em,
    [
        #v(3em)
        #table(
            columns: 2,
            align: (left, right),
            [ *Best agent* ], [ *Win-rate vs. greedy* ],
            [ REINFORCE ], [ 64.90% ],
            [ Monte Carlo ], [ 49.80% ],
            [ Q-Learning ], [ 40.10% ],
            [ DQN ], [ 27.60% ],
            [ DDQN ], [ 25.00% ],
        )
        - *REINFORCE* achieved *\~65% win-rate*
        - Tabular methods reached almost 50%
        - (D)DQN diverged
    ],
    [
        #align(center, image("images/reinforce_training.svg", width: 110%))
        #align(center)[#text(size: 9pt)[_REINFORCE win-rate throughout learning_]]
        #image("images/double_dqn_training.svg", width: 110%)
        #align(center)[#text(size: 9pt)[_DDQN win-rate throughout learning_]]
    ],
)


== Evaluation Against Human Players
#v(3em)
#align(
    [
        #table(
            columns: 4,
            align: (left, right, right, right),
            [ *Agent* ], [ *Human win-rate* ], [ *Games played* ], [ *Players* ],
            [ Greedy ], [ 65% ], [ 100 ], [ 1 ],
            [ REINFORCE ], [ *54.20%* ], [ 284 ], [ 10 ],
        )
    ],
    center,
)
#v(1em)
- The REINFORCE agent lowered the human win-rate significantly,
    performance was nearly even


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


== Conclusion and Future Work
- Policy gradient methods managed to learn a non-trivial policy
    even through the game's high level of stochasticity
- Future work:
    - Multi-agent environment (currently 1v1)
    - State representation: Using RNNs
    - Advanced algorithms: Proximal Policy Optimization (PPO),
        Soft Actor-Critic (SAC), MuZero
    - GUI for human evaluation


= Thank you for your attention \ #text([
    Questions?
], size: 12pt, weight: "medium", fill: black)


== Otázka opponenta \#01

Prostředí Prší představuje hru s neúplnou informací a významnou
náhodnou složkou. Jak byste změnil svůj přístup, pokud by agent měl k dispozici
úplnou informaci o stavu hry, tedy i karty soupeře a pořadí karet
v lízacím balíčku? Myslíte si, že by v takovém prostředí měly metody
DQN a DDQN větší šanci na úspěch?

#pagebreak()

- Metody (D)DQN by dostávaly stejný / ještě komplikovanější stav
- Znalost úplné informace o hře $=>$ dosud nepoužitelné algoritmy
    - Minimax, Alpha-Beta Pruning, ...
    - Monte Carlo Tree Search (AlphaZero)

#align(center, image("images/ab-pruning.png", height: 60%))

== Otázka opponenta \#02

Do jaké míry podle vás za neúspěchem DQN a DDQN stojí neúplná
informace o stavu hry a do jaké míry omezená výpočetní kapacita či
zvolená reprezentace stavu?

#pagebreak()

*Reprezentace stavu jako pro tabulkové metody*
- 1000 epizod, $epsilon = 0.1$, $gamma = 0.99$ learning rate = $5 dot 10^(-5)$, batch size = 32,\
    $2 times 1024$ skrytých neuronů, spouštěno lokálně
#align(center, image("images/dqn-small-training.png", width: 25em))
#align(center, image("images/dqn-small-eval.png", width: 25em))

#pagebreak()

*REINFORCE with Baseline*
- Monte Carlo odhad state-value funkce
/ * REINFORCE Baseline update*: $
delta <- G_t - hat(v)(S_t; bold(w))\
bold(w) <- bold(w) + alpha_w delta nabla hat(v)(S_t; bold(w))\
$
