#import "./template/template.typ": *
#import "@preview/glossarium:0.5.10": Gls, Glspl, gls, glspl, make-glossary, print-glossary, register-glossary
#import "./acronyms.typ": entry-list

#show: template.with(
    meta: (
        title: "Reinforcement Learning for Prší Card Game",
        author: (
            name: "Josef Vašata",
        ),
        // submission-date: datetime(year: 2012, month: 1, day: 21),
        submission-date: datetime.today(),
        // true for bachelor's thesis, false for master's thesis
        bachelor: true,
        faculty: "Information Technology",
        department: "Applied Mathematics",
        supervisor: "Ing. Daniel Vašata, Ph.D.",
    ),
    font: "New Computer Modern",

    // set to true if generating a PDF for print (shifts page layout, correctly aligns odd/even pages,...)
    print: false,

    abstract-en: [
        #lorem(40)

        #lorem(60)
    ],

    abstract-cz: [
        #lorem(40)

        #lorem(60)
    ],

    keywords-en: [
        machine learning, reinforcement learning, card games, Q-Learning,
        DQN, neural networks
    ],
    keywords-cz: [
        strojové učení, posilované učení, karetní hry, Q-Learning,
        DQN, neuronové sítě
    ],

    acknowledgement: [
        I would like to express my gratitude to my dear friends and family for their support,
        encouragement, and, naturally, the thousands of hands of Prší played together
        during my academic journey.
        I would also like to thank my supervisor, Ing. Daniel Vašata, Ph.D.,
        for his guidance and insight during the writing of this thesis.
    ],

    declaration: [
        I hereby declare that the presented thesis is my own work and that I have cited all sources of
        information in accordance with the Guideline for adhering to ethical principles when elaborating an
        academic final thesis.
        I acknowledge that my thesis is subject to the rights and obligations stipulated by the Act No.
        121/2000 Coll., the Copyright Act, as amended. In accordance with Section 2373(2) of Act No.
        89/2012 Coll., the Civil Code, as amended, I hereby grant a non-exclusive authorization (licence) to
        utilize this thesis, including all computer programs that are part of it or attached to it and all
        documentation thereof (hereinafter collectively referred to as the "Work"), to any and all persons
        who wish to use the Work. Such persons are entitled to use the Work in any manner that does not
        diminish the value of the Work and for any purpose (including use for profit). This authorisation is
        unlimited in time, territory and quantity.

        I declare that I have used AI tools during the preparation and writing of my thesis. I have verified
        the generated content. I confirm that I am aware that I am fully responsible for the content of the
        thesis.
    ],
    assignment: read("assignment.pdf", encoding: none),
)

#show: make-glossary
#register-glossary(entry-list)


#heading([Introduction], numbering: none)

The field of #gls("ai") has been gaining popularity in recent years, driven in
part by advances in #gls("ml"). #Gls("rl") is a subfield of #gls("ml") that aims to
teach computers how to behave without being explicitly shown
how to act, only telling the #gls("ai") agent how good the most recent thing
the agent did was by handing out a numerical reward for every action performed.

#Gls("rl") has shown superhuman performance in many board games and video games alike,
with AlphaZero @alphazero famously beating the best go players in the world and Agent57 @agent57
achieving superhuman performance in the whole Atari suite of games.
However, a less explored frontier remains: games with imperfect information.
In these settings, the "truth" of the game is hidden behind the back
of a card or the mind of an opponent, forcing an agent to reason under
deep uncertainty rather than just calculating a path through a known state.

In this thesis, we will focus on Prší, a cultural staple in the Czech Republic.
It is one of the many variants of the German card game Mau-Mau and is
not dissimilar to the world-famous card game Uno.
We'll use this unique setting to explore how both traditional and modern
#gls("rl") methods handle the hidden variables inherent to stochastic
environments like card games.


#heading([Goals], depth: 2, numbering: none, outlined: true)

The main goal of this thesis is to evaluate different #gls("rl") algorithms on their
performance in Prší and to create an agent that can defeat human opponents.

To achieve this, we will implement an environment capable of putting various
agents against each other in a 1v1 setting with 2 baseline opponents available.
The environment will also be able to update the opponent
on the fly for self-play compatibility.

We will use the implemented environment to train agents using both tabular
methods, such as Monte Carlo, and #gls("dl") based approaches, like
#gls("dqn", first: false).

Finally, after comparing the agents to a baseline to find the best one,
we will test its performance against human players and discuss the results.


= Reinforcement Learning <chapter:rl>

In this chapter, we introduce #gls("rl") and some of its formalisms and key concepts
-- such as #glspl("mdp", first: false) and value functions in @chapter:rl-intro.
Then we'll detail value-based methods (@chapter:value-methods) and
policy gradient methods (@chapter:policy-methods). These algorithms will
form the foundation for our Prší agents in @chapter:experiments.

== Introduction to Reinforcement Learning <chapter:rl-intro>

The main advantage of #gls("rl") over
#gls("sl") is its ability to perform well in tasks where #gls("sl") would simply
be too impractical, as gathering enough labeled data to train a competent model
isn't always feasible (e.g., manually labeling the "best" action in _every_ chess board state).
To circumvent this issue, #gls("rl") methods essentially create
their own training data by interacting with an environment.

Interactions with an environment give the agent feedback in the form
of numerical rewards, which the agents then try to maximize. These
rewards remove the need for labeled data by replacing the _"act as I was told"_
train of thought with _"act in a way that gets me as high of a reward
as possible"_.

The general idea of the environment is that the agent observes some state,
from which the agent takes an action. Performing this action is what
will result in a reward and will also move the agent into a different state,
where the loop begins anew. Formally, we'll model these environments as
#glspl("mdp", display: "Markov Decision Processes").

=== Markov Decision Process

#figure(
    image("images/mdp.png", width: 80%),
    caption: flex-caption(
        [MDP illustration],
        [Markov Decision Process @npfl139-lec01],
    ),
) <fig:mdp-loop>

To formalize any reinforcement learning problems, we model the environment as
a #gls("mdp", first: true). This model captures the interaction between
the agent and its environment through states, actions, transition probabilities,
and rewards. An illustration of an #gls("mdp") can be seen in @fig:mdp-loop.

A particular #gls("mdp") is defined as a quadruple $(cal(S), cal(A), p, gamma)$,
where $cal(S)$ is a set of states, $cal(A)(s)$ a set of actions that can be taken
in state $s in cal(S)$, $p$ the environment dynamics and $gamma in [0, 1]$
the discount factor.
//
#footnote(
    [ If $cal(S) "and" cal(A)$ are finite, we're talking about
        a finite #gls("mdp").],
)
//
Given a state $s$ and action $a$, the environment dynamics model
the probability of a next state $s'$ and reward $r$, formally denoted
$
    p(s', r mid(bar) s, a) = upright(P)(S_(t+1) = s', R_(t+1) = r mid(bar) S_t = s, A_t = a).
$
The reliance of dynamics only on $S_t$ and independence from $S_0, ..., S_(t-1)$
is called the Markov property.

Sometimes, #glspl("mdp") can also be defined as a quintuple with a reward probability $r$,
leaving us with the following transition and reward probabilities:
$
    p(S_(t+1) = s' mid(bar) S_t = s, A_t = a),
    #linebreak()
    r(R_(t+1) = r mid(bar) S_(t+1) = s', S_t = s, A_t = a).
$
What both these equivalent definitions tell us is that the reward is always tied
to the next state.

While the definition above certainly is useful, there are many tasks (such as mazes
or, fittingly, card games)
where even though the environment does have a state internally,
the agent doesn't know what the state looks like. In Prší, for example, no player
knows what cards the opponent has, even though a "full state" exists.
To model environments like these, we define the #gls("pomdp").

#Glspl("pomdp") are inherently similar to #glspl("mdp"), but they are
defined as a sextuple $(cal(S), cal(A), p, gamma, cal(O), o)$,
where $cal(O)$ is a set of observations and $o(O_(t+1) mid(bar) S_t, A_t)$ is an
observation model. We then give agents $O_t$ as input instead of $S_t$. An
illustration can be seen in @fig:pomdp-loop.~@Spaan2012 @Sutton2018 @npfl139-lec01

#figure(
    image("images/pomdp.png", width: 80%),
    caption: flex-caption(
        [POMDP illustration],
        [Partially Observable Markov Decision Process @npfl139-lec01],
    ),
) <fig:pomdp-loop>


=== Reward vs. Return

While we have said that the goal of the agent is to maximize the reward it gets,
it wasn't an entirely accurate formulation.
The goal of the agent is to maximize the _cumulative_
reward over the whole interaction. Let's imagine
a sequence of rewards after timestep $t$: $R_(t+1), R_(t+2), R_(t+3), ...$

Our goal will be to maximize the *return* $G_t$, which can in its
simplest form be defined as
$
    G_t = R_(t+1) + R_(t+2) + R_(t+3) + ... + R_T
$
where $T$ is the final timestep. This approach works well for tasks where such
a final timestep can be found, resulting in many subsequences of interactions,
which we call episodes. Tasks that can be split into episodes are called episodic.
Episodes always end when a _terminal state_ is reached. The interaction is
then restarted from a starting state. Environments can have multiple starting
and terminal states.
#footnote([$cal(S)^+$ is sometimes used to signify episodic tasks as
    "$cal(S)$ with terminal states".])

The alternative are continuing tasks, where an interaction can theoretically go
on forever. In this case, $G_t$ as we defined before could potentially be unbounded
and grow to infinity. To prevent this, a discount factor $gamma < 1$ can be used.
We then define $G_t$ as follows:
$
    G_t = R_(t+1) + gamma R_(t+2) + gamma^2 R_(t+3) + ...
    = sum_(k=0)^infinity gamma^k R_(t+1+k).
$
In episodic tasks, if we introduce an absorbing state which can't be transitioned
out of and gives a reward of 0, we can use this formula for both episodic and continuing tasks.
We can also use $G_t = sum_(k=0)^(T-t-1) gamma^k R_(t+1+k)$
and allow for $T = infinity$ or $gamma = 1$ (never both).
Fixing $gamma < 1$ can however be useful even in episodic tasks, as it
serves to weight immediate rewards more heavily than distant ones. This
encourages the agent to seek the fastest path to victory.

With this definition of the return $G_t$, we can now finally formalize the goal
of the agent, that being maximization of $EE[G_t]$ (or specifically $EE[G_0]$
for episodic tasks).~@Sutton2018 @npfl139-lec01

=== (Action-)Value function

Value functions are used to evaluate policies.
The agent's policy $pi$ defines the behaviour of the agent by determining
which action the agent will select in a given state. For a deterministic
policy, the action is chosen directly as $a = pi(s)$. In case of
stochastic policies, $pi(a mid(bar) s)$ denotes the probability of
selecting action $a$ given a state $s$.

The value function $v_pi (s)$ (sometimes also called the state-value function)
for a policy $pi$ gives us the expected
return if the agent starts in state $s$ and then follows $pi$ in each step.
Formally, we can define it for #glspl("mdp") as
$
    v_pi (s) = EE_pi [G_t mid(bar) S_t = s]
    = EE_pi lr([sum_(k=0)^infinity gamma^k R_(t+1+k) mid(bar) S_t = s])
$
where $EE_pi [dot]$ denotes the expected value given that the agent follows
policy $pi$ to select actions, and $t$ can be any timestep.

We will also define the action-value function $q_pi (s, a)$, which denotes
the expected return after taking action $a$ in state $s$ and following $pi$
afterwards:
$
    q_pi (s, a) = EE_pi [G_t mid(bar) S_t = s, A_t = a]
    = EE_pi lr([sum_(k=0)^infinity gamma^k R_(t+1+k) mid(bar) S_t = s, A_t = a]).
$\

Finally, we define the optimal value function $v_*$ and the optimal
action-value function $q_*$ as those which have the highest values across
all possible policies. Formally:
$
    v_* = max_pi v_pi (s)\
    q_* = max_pi q_pi (s, a)
$
for all $s in cal(S)$ and $a in cal(A)(s)$. Any policy $pi_*$ with
$v_pi_* = v_*$ is called the optimal policy, as there can be more than one.

A fundamental property of value functions is that they satisfy
recursive relationships. For any policy $pi$, the value of a state
can be decomposed into the immediate reward plus the discounted
value of the expected next state. This is known as the *Bellman equation*
for $v_pi$:
$
    v_pi (s) = EE[G_t mid(bar) S_t = s]
    = sum_a pi(a mid(bar) s) sum_(s', r) p(s', r mid(bar) s, a) [r + gamma v_pi (s')].
$
where $a in cal(A)(s)$ and $s, s' in cal(S)$. We also define the *Bellman
optimality equation* for $v_*$, which expresses the fact that the value of a
state under an optimal policy must equal the expected return for the best action
from that state:
$
    v_* (s) = max_a q_* (s, a)
    = max_a sum_(s', r) p(s', r mid(bar) s, a) [r + gamma v_* (s')].
$
The Bellman optimality equation represents a system of equations -- one
for each state -- the solution to which is the optimal value function $v_*$.
If the environment dynamics $p(s', r mid(bar) s, a)$ are known, this system
can be solved using classical Dynamic Programming algorithms such as
*Value Iteration* or *Policy Iteration*.

However, for many complex tasks, including Prší, the
transition probabilities are either unknown or too complex to compute.
In these cases, we must rely on model-free reinforcement learning methods.
These methods allow the agent to learn the optimal policy through direct
interaction with the environment without requiring explicit knowledge of
the dynamics.~@Sutton2018 @npfl139-lec02

=== Exploration vs. Exploitation

Since the agent will be learning through interactions with the environment,
a fundamental question arises: which policy should be followed during the
training process? Behaving in a strictly greedy fashion (_exploitation_) can
be counterproductive, as our action-value function estimate isn't accurate and
could leave the agent stuck in local optima.
On the other hand, behaving completely randomly (_exploration_) isn't
the solution either, as that would make it impossible for the agent to estimate
$q_pi$ for strategic policies.

#let eps_soft = $epsilon/(|cal(A)(s)|)$

To solve this dilemma, we will introduce soft policies, that is policies $pi$
where $pi(a mid(bar) s) > 0$ for all $s in cal(S)$ and all $a in cal(A)(s)$.
In particular, we will focus on $epsilon$-greedy policies ($epsilon < 1$), which are
policies where the agent chooses the greedy action with a probability
$1 - epsilon$ and a random action with a probability $epsilon$. This gives
all non-greedy actions the probability of #eps_soft and the greedy action
probability $1 - epsilon + #eps_soft$, as the greedy action can still be
randomly chosen. It is essentially the greediest of all $epsilon$-soft policies,
meaning policies where all actions must have at least this
probability.

There is one final improvement we can make. In early training, the agent
(probably) has very bad estimates of $q_pi$. Therefore, it doesn't make sense
to behave greedily too often. We can start with a high $epsilon$ like 0.5 and
slowly decay it after each step in the environment by multiplying it by a value
like 0.99 until it reaches some minimum threshold (or even 0). This lets us
collect a lot of different samples before gradually shifting to
a more greedy policy.~@Sutton2018

== Value-Based Methods <chapter:value-methods>

With the theoretical foundation established, we can now examine the concrete
#gls("rl") algorithms that we'll be implementing in @chapter:experiments.
We begin with *value-based methods*,
which, as the name implies, focus on estimating the optimal
action-value function $q_*$.

The core idea of these approaches is that if we can accurately predict
the long-term value of every action in every state, the task of finding
an optimal policy becomes trivial. By simply selecting the action with
the highest estimated value -- the greedy approach with respect
to the value function -- the agent can derive its behavior without ever having
to explicitly learn a separate policy function. Once we have estimated $q_*$
as $hat(q)_*$ during training, we'll simply select actions deterministically
by using the policy
$
    pi (s) = argmax_a hat(q)_* (s, a).
$

=== Monte Carlo

#lorem(70)

=== Q-Learning

#lorem(70)

=== Deep Q-Network

#lorem(70)

== Policy Gradient Methods <chapter:policy-methods>

=== REINFORCE

#lorem(70)

#lorem(100)

// Maybe PPO as well


= Implementing an Environment for Prší <chapter:environment>

== The Rules of Prší

#lorem(100)

== Designing a Reinforcement Learning Environment

#lorem(100)

= Experiments <chapter:experiments>

== Evaluating Agent Performance

#lorem(100)

== Comparison of Implemented Agents

#lorem(100)

== Performance Against Human Players

#lorem(100)


= Future Work

#lorem(100)


#heading([Conclusion], numbering: none)

#lorem(100)


#bibliography("bibliography.bib")


// all h1 headings from here on are appendices
#show: start-appendix

= Acronyms

#print-glossary(
    entry-list,
    show-all: false,
    user-print-glossary: acronym-table,
    disable-back-references: true,
)
