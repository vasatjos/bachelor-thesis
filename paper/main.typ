#import "./template/template.typ": *
#import "@preview/glossarium:0.5.10": Gls, Glspl, gls, glspl, make-glossary, print-glossary, register-glossary
#import "@preview/algo:0.3.6": algo, code, comment, d, i
#import "./acronyms.typ": entry-list

#let human_greedy_wins = 65
#let human_greedy_games = 100
#let human_greedy_rate = calc.round((human_greedy_wins / human_greedy_games) * 100, digits: 1)

#let human_rl_wins = 72
#let human_rl_games = 121
#let human_rl_rate = calc.round((human_rl_wins / human_rl_games) * 100, digits: 1)


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

    two-page-abstract: true,

    abstract-en: [
        This thesis investigates the application of reinforcement learning to
        Prší, a stochastic, imperfect-information Czech card game. A custom Python
        environment was developed to evaluate four algorithms: Monte Carlo,
        Q-Learning, Deep Q-Network (DQN), and REINFORCE. To handle the environment's
        complexity, tabular methods relied on abstracted state representations,
        while the deep learning agents processed full one-hot encoded observations.

        While deep value-based methods (DQN) struggled to converge within training
        constraints, the best tabular method runs achieved a win rate of
        just under 50% against a greedy baseline.
        The REINFORCE policy gradient algorithm emerged as the top
        performer, achieving a ~65% win rate against the greedy baseline. In a final
        human evaluation across #human_rl_games games, the REINFORCE agent successfully held
        human players to a ~#human_rl_rate% win rate -- a tangible reduction from
        the 65% human win rate against the baseline. These results demonstrate that
        policy gradient methods can learn somewhat competitive strategies in
        complex card games using purely rule-based interactions.
    ],

    abstract-cz: [
        Tato práce zkoumá aplikaci posilovaného učení (reinforcement learning)
        na Prší, stochastickou karetní hru s neúplnou informací. Pro účely
        vyhodnocení čtyř algoritmů -- Monte Carlo, Q-Learning, Deep Q-Network (DQN)
        a REINFORCE -- bylo vytvořeno prostředí v jazyce Python.
        Pro zvládnutí složitosti prostředí
        se tabulkové metody spoléhaly na abstrahované reprezentace stavů,
        zatímco agenti
        využívající hluboké učení zpracovávali kompletní pozorování kódovaná metodou one-hot.

        Zatímco hluboké metody založené na odhadu $q$ funkce (DQN) měly potíže
        s konvergencí v rámci tréninkových omezení, nejlepší běhy
        tabulkových metod dosáhly míry výher těsně pod
        50 % proti referenční hladové (greedy) strategii.
        Algoritmus REINFORCE se ukázal jako nejúspěšnější a dosáhl
        přibližně 65% míry výher proti hladové referenční strategii.
        V závěrečném vyhodnocení proti lidem agent úspěšně udržel míru výher
        lidských hráčů na ~#human_rl_rate % -- což představuje
        znatelný pokles oproti 65% úspěšnosti lidí proti referenčnímu agentovi.
        Tyto výsledky dokazují, že metody posilovaného učení se dokážou naučit
        do jisté míry konkurenceschopné
        strategie v komplexních karetních hrách pouze na
        základě interakcí s pravidly prostředí.
    ],

    keywords-en: [
        machine learning, reinforcement learning, Q-Learning,
        DQN, neural networks, imperfect information games, card games, Prší
    ],
    keywords-cz: [
        strojové učení, posilované učení, Q-Learning,
        DQN, neuronové sítě, hry s neúplnou informací, karetní hry, Prší
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
with AlphaZero~@alphazero famously beating the best go players in the world and Agent57~@agent57
achieving superhuman performance in the whole Atari suite of games.
However, a less explored frontier remains: games with imperfect information.
In these settings, the "truth" of the game is hidden behind the back
of a card or the mind of an opponent, forcing an agent to reason under
deep uncertainty rather than just calculating a path through a known state.

In this thesis, we will focus on Prší~@prsi, a cultural staple in the Czech Republic.
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

Unless otherwise noted, the mathematical notation, foundational definitions,
and general algorithmic frameworks presented in this chapter closely follow
@Sutton2018, with some minor simplifications
being inspired by @npfl139-lec01 and @npfl139-lec02.

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
        [Markov Decision Process],
        [Markov Decision Process~@npfl139-lec01],
    ),
) <fig:mdp-loop>

To formalize any #gls("rl") problem, we model the environment as
a #gls("mdp", first: true). This model captures the interaction between
the agent and its environment through states, actions, transition probabilities,
and rewards. An illustration of an #gls("mdp") can be seen in @fig:mdp-loop.

A particular #gls("mdp") is defined as a quadruple $(cal(S), cal(A), p, gamma)$,
where $cal(S)$ is the set of states, $cal(A)(s)$ the set of actions that can
be taken in state $s in cal(S)$, $p$ the environment dynamics and
$gamma in [0, 1]$ the discount factor.
//
#footnote([
    If $cal(S) "and" cal(A)$ are finite, we're talking about
    a finite #gls("mdp").
])
//
Given a state $s$ and action $a$, the environment dynamics model
the probability of a next state $s'$ and reward $r$, formally denoted
$
    p(s', r mid(bar) s, a) = upright(Pr){S_(t+1) = s', R_(t+1) = r mid(bar) S_t = s, A_t = a}.
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
To model environments like these, we define the #gls("pomdp")~@KAELBLING199899 @Spaan2012.

#Glspl("pomdp") are inherently similar to #glspl("mdp"), but they are
defined as a sextuple
//
#footnote([
    #gls("pomdp") also has a similar alternate definition to #gls("mdp").
])
//
$(cal(S), cal(A), p, gamma, cal(O), o)$,
where $cal(O)$ is the set of observations and $o(O_(t+1) mid(bar) S_t, A_t)$ is
the observation model. We then give agents $O_t$ as input instead of $S_t$. An
illustration can be seen in @fig:pomdp-loop.

#figure(
    image("images/pomdp.png", width: 80%),
    caption: flex-caption(
        [Partially Observable Markov Decision Process],
        [Partially Observable Markov Decision Process~@npfl139-lec01],
    ),
) <fig:pomdp-loop>


=== Reward vs. Return

While we have said that the goal of the agent is to maximize the reward it gets,
it wasn't an entirely accurate formulation.
The goal of the agent is to maximize the _cumulative_
reward over the whole interaction.

Let's take
a sequence of rewards after timestep $t$: $R_(t+1), R_(t+2), R_(t+3), ...$
Our goal will be to maximize the _return_ $G_t$, which can in its
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
#footnote([
    $cal(S)^+$ is sometimes used to signify episodic tasks as
    "$cal(S)$ with terminal states".
])

The alternative are continuing tasks, where an interaction can theoretically go
on forever. In this case, $G_t$ as we defined before could potentially be unbounded
and grow to infinity. To prevent this, a discount factor $gamma < 1$ can be used.
We then define $G_t$ as follows:
$
    G_t = R_(t+1) + gamma R_(t+2) + gamma^2 R_(t+3) + ...
    = sum_(k=0)^infinity gamma^k R_(t+1+k).
$
To use this formula in episodic tasks as well, we can introduce an absorbing
state which can't be transitioned out of and gives a reward of 0.
We can also use $G_t = sum_(k=0)^(T-t-1) gamma^k R_(t+1+k)$
and allow for $T = infinity$ or $gamma = 1$ (never both).
Fixing $gamma < 1$ can however be useful even in episodic tasks, as it
serves to weigh immediate rewards more heavily than distant ones. This
encourages the agent to seek the fastest path to victory.

With this definition of the return $G_t$, we can now finally formalize the goal
of the agent, that being maximization of $EE[G_t]$ (or specifically $EE[G_0]$
for episodic tasks).

=== (Action-)Value Function

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
value of the expected next state. This is known as the _Bellman equation_
for $v_pi$:
$
    v_pi (s) = EE[G_t mid(bar) S_t = s]
    = sum_a pi(a mid(bar) s) sum_(s', r) p(s', r mid(bar) s, a) [r + gamma v_pi (s')]
$
where $a in cal(A)(s)$ and $s, s' in cal(S)$. We also define the _Bellman
optimality equation_ for $v_*$, which expresses the fact that the value of a
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
_Value Iteration_ or _Policy Iteration_.

However, for many complex tasks, including Prší, the
transition probabilities are either unknown or too complex to compute.
In these cases, we must rely on model-free #gls("rl") methods.
These methods allow the agent to learn the optimal policy through direct
interaction with the environment without requiring explicit knowledge of
the dynamics.

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
a more greedy policy.

== Value-Based Methods <chapter:value-methods>

With the theoretical foundation established, we can now examine the concrete
#gls("rl") algorithms that we'll be implementing in @chapter:experiments.
We begin with _value-based methods_,
which, as the name implies, focus on approximating the optimal
action-value function $q_*$ through a learned estimate $Q$.

The core idea of these approaches is that if we can accurately predict
the long-term value of every action in every state, the task of finding
an optimal policy becomes trivial. By simply selecting the action with
the highest estimated value -- the greedy approach with respect
to the action-value function -- the agent can derive its behaviour without ever
having to explicitly learn a separate policy function. Once we have
estimated $q_*$ as $Q$ during training, we'll simply select actions
deterministically by using the policy
$
    pi (s) = argmax_a Q (s, a).
$

=== Monte Carlo

#Gls("mc") methods are a fundamental class of #gls("rl") algorithms
that learn value functions directly from raw episodes of experience.
Unlike dynamic programming approaches, #gls("mc") methods do not require prior
knowledge of the environment's transition probabilities or reward dynamics.
Instead, they estimate the action-value function $q_(pi)(s, a)$ by
averaging the sample returns observed during actual interactions with the
environment.

Because #gls("mc") methods rely on calculating the final return $G_t$,
they require the experience to be divided into well-defined
episodes that eventually terminate. This makes them a natural fit for card
games like Prší, where every played game represents a single episode
that strictly concludes with a terminal state where one player wins.

To estimate the action values, an agent plays out an entire episode using
its current policy. Once a terminal state is reached, the agent
looks back at the trajectory of states, actions, and rewards to calculate
the true return for each step. Because a state-action pair might be visited
multiple times within the same episode, we must decide which visits to learn from.
_First-visit_ #gls("mc") averages only the returns following the first time a
state-action pair is visited in an episode, whereas _every-visit_ #gls("mc")
averages the returns following all visits. While every-visit #gls("mc")
has states which may not be independent, both of these methods still
converge to the true value function.

#let monte_carlo_update_rule = weight => {
    $
        Q(S_t, A_t) <- Q(S_t, A_t) + weight [G_t - Q(S_t, A_t)]
    $
}
#let sa_pair_visits = $C(S_t, A_t)$

To calculate the average return over multiple episodes, the most straightforward
approach is to use a simple average of observed samples. Let #sa_pair_visits
be the number of times action $A_t$ has been taken in state $S_t$. The update
rule then is:
$
    #monte_carlo_update_rule($1 / #sa_pair_visits$).
$
However, this is not optimal in the sense that it weighs the very oldest
observed return with the same weight as the most recent one. This is reasonable
if the policy never changes, but when it does, it makes more sense to give more
weight to recently observed values. We can do this by using an exponential
moving average with the parameter $alpha$, where we get
$
    #monte_carlo_update_rule($alpha$)
$
as the update rule.

To guarantee that the estimates for all state-action pairs converge to their
true values, the agent must continuously explore the environment. In theoretical
settings, this is often handled by the assumption of _exploring starts_, where
every episode is forced to begin with a randomly selected state and action.
In many environments, enforcing a random initial action is unnatural and
selecting a random state may not even be possible at all.
However, the inherent randomness of shuffling and dealing cards
in Prší serves a highly similar purpose: it ensures the agent is constantly exposed
to a vast distribution of starting states without any manual intervention.

Because random starting states alone do not guarantee exploration in the deeper
stages of an episode, nor will they result in a random first action,
our agent still utilizes an $epsilon$-greedy policy.
This combination ensures that the agent sufficiently explores the state space while
simultaneously optimizing its behaviour. A simple #gls("mc") algorithm with an
$epsilon$-greedy policy using $alpha$ as the update step can be seen
in @alg:mc-control.

#figure(
    algo(
        title: [Monte Carlo Control],
        parameters: ([episodes], $epsilon$, $gamma$, $alpha$),
        line-numbers: false,
    )[
        #let StAt = $(S_t, A_t)$

        Initialize $Q(s, a) <- 0$ for all $s in cal(S), a in cal(A)(s)$\
        Loop for each episode:#i\
        Generate trajectory $S_0, A_0, R_1, S_1, A_1, R_2, ..., S_(T-1), A_(T-1), R_T$ using $epsilon$-greedy policy derived from $Q$\
        $G <- 0$\

        Loop for $t = T-1$ down to $0$:#i\
        $G <- R_(t+1) + gamma G$\
        If #StAt not in $(S_0, A_0), ..., (S_(t-1), A_(t-1))$:#i\
        $Q #StAt <- Q #StAt + alpha [G - Q #StAt]$
    ],
    caption: flex-caption(
        [$epsilon$-greedy Monte Carlo Control],
        [First-visit Monte Carlo Control with $epsilon$-greedy exploration~@Sutton2018],
    ),
    kind: "algo",
    supplement: "Algorithm",
) <alg:mc-control>

=== Q-Learning

While #gls("mc") methods are intuitive and effective, they suffer from a
significant limitation: they must wait until the end of an episode to observe
the final return $G_t$ before any learning can occur. In environments with
long episodes, this delays the update process, even if many rewards were
already observed throughout the episode. Furthermore, if an episode
is never guaranteed to terminate, standard #gls("mc") methods cannot be used
at all.

Q-Learning~@Watkins1992 elegantly bypasses this issue by utilizing #gls("td") learning.
Instead of waiting for the true episodic return, #gls("td") methods update their
value estimates based in part on other learned estimates -- a process known as
_bootstrapping_.

In Q-Learning, the agent takes an action $A_t$ in state $S_t$, observes
the immediate reward $R_(t+1)$ and the next state $S_(t+1)$, and
immediately performs an update. To do this, it replaces the full return $G_t$
used in #gls("mc") with the _#gls("td") target_, which estimates the remainder of the
return by assuming the agent will take the optimal action from the
next state onward.

This brings us to the second major distinction from #gls("mc") methods:
Q-Learning is an _off-policy_ algorithm. It separates the _behaviour policy_
(the policy used to interact with the environment and gather data,
such as $epsilon$-greedy) from the _target policy_ (the policy being
learned and evaluated). The algorithm learns the optimal action-value
function $q_*$ directly, regardless of the agent's exploratory actions.

The Q-Learning update rule is derived directly from the Bellman Optimality
Equation and is defined as:
$
    Q(S_t, A_t) <- Q(S_t, A_t) + alpha [R_(t+1) + gamma max_a Q(S_(t+1), a) - Q(S_t, A_t)]
$
where the term $R_(t+1) + gamma max_a Q(S_(t+1), a)$ acts as our
bootstrapped #gls("td") target.

What this update rule states is that the agent modifies the estimate to account
for the new observed reward and assumes greedy behaviour will be followed
afterwards. This can actually result in worse results during training, as
the assumption of greedy behaviour is not correct under the $epsilon$-greedy
policy. If the evaluation uses $epsilon = 0$ however, the agent's estimates
for $Q$ will be "more correct" than the #gls("mc") estimates, which have
learned the action-value function only for the $epsilon$-greedy case.

#figure(
    algo(
        title: [Q-Learning],
        parameters: ([episodes], $alpha$, $epsilon$, $gamma$),
        line-numbers: false,
    )[
        Initialize $Q(s, a)$ arbitrarily for all $s in cal(S), a in cal(A)(s)$\
        except that $Q("terminal", dot) = 0$\
        Loop for each episode:#i\
        Initialize $S$\
        Loop for each step of episode:#i\
        Choose $A$ from $S$ using $epsilon$-greedy policy derived from $Q$\
        Take action $A$, observe $R, S'$\
        $Q(S, A) <- Q(S, A) + alpha [R + gamma max_a Q(S', a) - Q(S, A)]$\
        $S <- S'$#d\
        until $S$ is terminal\
    ],
    caption: flex-caption(
        [Q-Learning],
        [Q-Learning: Off-policy TD control algorithm~@Sutton2018],
    ),
    kind: "algo",
    supplement: "Algorithm",
) <alg:q-learning>

=== Deep Q-Network

A fundamental issue with the methods presented so far was that they were so-called
_tabular methods_. $Q$ was just a table with a field for each state-action pair.
This is not ideal for many environments, including Prší, as different states
may share a large part of information. Although specific state representations
for Prší will be discussed in @chapter:environment and @chapter:experiments,
it is clear that many states are fundamentally similar. For instance, two
game states might differ by only a single played card, yet their underlying
strategic values are nearly identical.
Tabular methods fail to capture this shared structure, treating each state in isolation.

To solve this problem of states being "too independent", we'll look at our
approximations not as tables, but functions parametrized by a weight vector
$bold(w) in RR^d$. This will allow the agent to generalize learned
experiences across many similar states. We'll denote our estimates for
the value function and action-value function as follows:
$
    hat(v)(s; bold(w))\
    hat(q)(s, a; bold(w)).
$
In the field of #gls("rl"), this approach is fittingly
called _function approximation_.

To find the optimal weight vector $bold(w)$, we must define an objective
function to minimize. In tabular methods, an update to one state does not
affect any other state, making it theoretically possible to learn the exact
value of every state. With function approximation, however, updating the weight
vector to improve the estimate for one state alters the estimates for
many others. Since we cannot achieve zero error for all states simultaneously,
we must specify which states we care about most.

We do this by weighting the errors according to a state distribution
$mu(s) >= 0$, $sum_s mu(s) = 1$. The most logical choice for this distribution
is one which represents the fraction of time the agent spends in state $s$.
We can then define the Mean Squared Value Error $overline(upright(V E))$
for the state-value function as:
$
    overline(upright(V E))(bold(w)) =
    sum_(s in cal(S)) mu(s) [v_pi (s) - hat(v)(s; bold(w))]^2.
$\

In practice, calculating this exact sum is impossible because the true state
distribution $mu(s)$ and the true values $v_pi (s)$ are unknown,
and the state space is typically too vast to iterate over. However, as the
agent interacts with the environment, it naturally visits states with frequencies
that approximate $mu(s)$. This allows us to bypass the intractable
full sum over the state space. Instead, we can estimate the expected
error by sampling transitions from the agent's collected experience
and adjusting the weights using #gls("sgd").

Then, we can either use the real return for gradient #gls("mc") algorithms,
or, just as in tabular Q-Learning, we can
substitute the unknown true return with a bootstrapped #gls("td") target.

This leads us to #gls("dqn", first: true)~@dqn2015, which brought 2
major innovations in using #glspl("nn") for approximating $Q$ values:
a separate _target network_ and _experience replay_.

To stabilize training with non-linear #glspl("nn"), #gls("dqn")
first employs experience replay. Because sequential game states are highly
correlated, training directly on consecutive interactions violates the standard
machine learning assumption of independent data. It can also lead to forgetting
past experiences. Instead, the agent stores transitions
$(S_t, A_t, R_(t+1), S_(t+1))$ in a (finite) memory buffer $D$ and uniformly samples
random mini-batches for training, effectively breaking temporal correlations.
Using these randomized batches, the primary network updates its weights using
#gls("sgd") (or possibly other #gls("nn") optimization methods like Adam~@Adam)
to minimize the Mean Squared Error against the #gls("td") target:
$
    L(bold(w)) = EE lr(
        [
            (R_(t+1) + gamma max_(a') hat(q)(S_(t+1), a'; bold(w))
                - hat(q)(S_t, A_t; bold(w)))^2
        ]
    ).
$\

However, calculating the target $hat(q)(S_(t+1), a'; bold(w))$ using the exact
same weight vector $bold(w)$ that is actively being updated
creates a problem, often described as chasing a moving target.
To solve this, #gls("dqn") introduces a separate target network
parameterized by $bold(w)^-$. This secondary network computes the target
value $R_(t+1) + gamma max_(a') hat(q)(S_(t+1), a'; bold(w)^-)$.
Its weights are held completely fixed during backpropagation and are only
periodically synchronized with the primary network $bold(w)$ every $C$ timesteps,
providing a stationary objective that prevents the learning process
from diverging. The actual loss function optimized at each step thus becomes:
$
    L(bold(w)) = EE lr(
        [
            (R_(t+1) + gamma max_(a') hat(q)(S_(t+1), a'; bold(w)^-)
                - hat(q)(S_t, A_t; bold(w)))^2
        ]
    ).
$

=== Double Deep Q-Network
A well-known issue with both standard Q-Learning and #gls("dqn") is the overestimation
of action values, often referred to as maximization bias. This occurs because
the #gls("td") target uses a maximum operator over estimated values, effectively
using the same network to both select the best action and evaluate its worth.
This leads to the evaluation being very good, as that's the reason the action
was selected in the first place. In tabular settings, this overly optimistic
behaviour is solved by Double Q-Learning~@Hasselt2010, which
decouples the selection and evaluation steps by maintaining two completely
separate $Q$ tables. However, maintaining two independent $Q$ tables
would normally double the memory requirements.

Fortunately, #gls("ddqn")~@Hasselt2016 offers an
elegant solution that acts essentially as a free upgrade. Since standard
#gls("dqn") already maintains two separate networks -- the primary network
$bold(w)$ and the target network $bold(w)^-$ -- #gls("ddqn") simply
leverages them to
decouple the maximization step without needing any additional parameters. The
primary network is used to select the greedy action in the next state, and the
target network is used to evaluate the value of that specific action.
The modified target for the loss function becomes:
$
    R_(t+1) + gamma hat(q)(S_(t+1), argmax_(a') hat(q)(S_(t+1), a'; bold(w)); bold(w)^-).
$
By separating the action selection from the action evaluation, #gls("ddqn")
prevents positive bias from compounding, leading to significantly more
stable training.

#figure(
    algo(
        title: [Deep Q-Learning with Experience Replay],
        parameters: ([capacity $N$], [update frequency $C$], $epsilon$, $gamma$),
        line-numbers: false,
    )[
        Initialize replay memory $D$ to capacity $N$\
        Initialize action-value function $hat(q)$ with random weights $bold(w)$\
        Initialize target action-value function with weights $bold(w)^- <- bold(w)$\
        Loop for each episode:#i\

        Initialize state $S_0$\
        Loop for each step $t$ of episode:#i\

        Choose $A_t$ from $S_t$ using $epsilon$-greedy policy derived from $hat(q)(dot, dot; bold(w))$\
        Take action $A_t$, observe reward $R_(t+1)$ and next state $S_(t+1)$\
        Store transition $(S_t, A_t, R_(t+1), S_(t+1))$ in $D$\
        Sample random mini-batch of transitions $(S_j, A_j, R_(j+1), S_(j+1))$ from $D$\
        #comment([Calculate TD target $Y_j$ for each transition in the mini-batch], inline: true)\

        If $S_(j+1)$ is a terminal state:#i\
        $Y_j <- R_(j+1)$#d\
        Else:#i\
        $Y_j <- R_(j+1) + gamma max_(a') hat(q)(S_(j+1), a'; bold(w)^-)$#d\

        Perform a gradient descent step on $(Y_j - hat(q)(S_j, A_j; bold(w)))^2$ with respect to $bold(w)$\
        Every $C$ steps, synchronize target network: $bold(w)^- <- bold(w)$\
        $S_t <- S_(t+1)$#d\

        until $S_t$ is terminal\
    ],
    caption: flex-caption(
        [Deep Q-Network],
        [Deep Q-Learning with experience replay and target networks~@dqn2015],
    ),
    kind: "algo",
    supplement: "Algorithm",
) <alg:dqn>

== Policy Gradient Methods <chapter:policy-methods>

We now look at _policy gradient methods_, which take a fundamentally different
approach to the value-based methods.
Instead of relying on a value function to dictate behaviour, these methods
parameterize the policy itself directly. Let $bold(theta) in RR^d$ be
the parameter vector for our policy. We can then denote the probability
that the agent takes action $a$ in state $s$ as a parameterized mathematical
function:
$
    pi (a mid(bar) s; bold(theta)) =
    upright(Pr){A_t = a mid(bar) S_t = s, bold(theta)}.
$\

Learning the policy directly offers several distinct advantages. Most notably
for environments with imperfect information like Prší, it allows the agent to
learn any distribution over actions. The agent can then choose actions
by sampling from this distribution. A value-based agent can only learn
one-hot (or soft) policies, while using policy gradient methods
allows for a stochastic policy, which can be beneficial in certain environments.

To optimize the policy parameters, we define a scalar performance measure
$J(bold(theta))$, which represents the expected return. The parameters are
then updated by performing gradient ascent to maximize this objective:
$
    bold(theta)_(t+1) = bold(theta)_t + alpha nabla J(bold(theta)_t),
$
where $alpha$ is the step-size parameter and $nabla J(bold(theta)_t)$ is the
policy gradient.

=== REINFORCE

To update the parameters using gradient ascent, we need a way to calculate the
gradient of our performance measure. The Policy Gradient Theorem
establishes an analytical gradient for $J(bold(theta))$ that does not depend
on the derivative of the environment's unknown state distribution. It proves
that the gradient is proportional to the expected value of the action-value
function multiplied by the gradient of the natural logarithm of the policy:
$
    nabla J(bold(theta)) prop
    EE_pi [q_pi (S_t, A_t) nabla ln pi (A_t mid(bar) S_t; bold(theta))].
$

Because this is an expectation over the policy $pi$, we can approximate it by
sampling trajectories. The REINFORCE algorithm~@Williams1992 is a
Monte Carlo policy gradient method that relies on this exact formulation.

Since REINFORCE is a Monte Carlo method, it uses the true sampled return $G_t$
as an unbiased estimate of $q_pi (S_t, A_t)$. The parameter update rule for a
given timestep $t$ becomes:
$
    bold(theta)_(t+1) =
    bold(theta)_t + alpha gamma^t G_t nabla ln pi (A_t mid(bar) S_t; bold(theta)_t).
$

Intuitively, this update rule increases the probability of actions that resulted
in a high return $G_t$, and decreases the probability of actions that resulted
in a low or negative return. The $ln pi$ term ensures that the update is scaled
inversely by the action's current probability, preventing frequently taken
actions from dominating the gradient simply because they appear more often, as
$nabla ln pi = (nabla pi) / pi$.


#figure(
    algo(
        title: [REINFORCE],
        parameters: ([step size $alpha > 0$],),
        line-numbers: false,
    )[
        Initialize policy parameter $bold(theta) in RR^d$\
        Loop for each episode:#i\
        Generate an episode $S_0, A_0, R_1, ..., S_(T-1), A_(T-1), R_T$ following $pi(dot mid(bar) dot, bold(theta))$\
        Loop for each step of the episode $t = 0, 1, ..., T - 1$:#i\
        $G <- sum_(k=t+1)^T gamma^(k-t-1) R_k$\
        $bold(theta) <- bold(theta) + alpha gamma^t G nabla ln pi(A_t mid(bar) S_t, bold(theta))$#d\
        #d
    ],
    caption: flex-caption(
        [REINFORCE],
        [REINFORCE: Monte-Carlo Policy-Gradient Control @Sutton2018],
    ),
    kind: "algo",
    supplement: "Algorithm",
) <alg:reinforce>

=== REINFORCE with Baseline

While the standard REINFORCE algorithm provides an unbiased estimate of the
policy gradient, it suffers from a significant practical limitation: high
variance. Because the parameter updates are scaled directly by the raw episodic
return $G_t$, a trajectory with exceptionally high or low stochastic rewards can
cause massive, destabilizing shifts in the policy parameters.

To reduce this variance, we can generalize the policy gradient theorem by
subtracting a baseline, denoted as $b(S_t)$, from the return. The baseline can
be any function or random variable, so long as it does not vary with the
selected action $a$. Mathematically, subtracting a baseline leaves the expected
value of the update unchanged, but it drastically reduces the variance of the
sampled gradients. The updated policy gradient formulation becomes:
$
    nabla J(bold(theta)) prop
    EE_pi [(q_pi (S_t, A_t) - b(S_t)) nabla ln pi (A_t mid(bar) S_t; bold(theta))].
$\

A natural and highly effective choice for the baseline is an estimate of the
state-value function, $hat(v)(S_t; bold(w))$. By using the state-value
estimate, the update rule effectively evaluates actions based on their
_advantage_ rather than their raw return. If an action yields a return $G_t$
that is greater than the expected baseline $hat(v)(S_t; bold(w))$, the
probability of that action is increased. If the return is lower than expected,
the probability is decreased.

To implement REINFORCE with a state-value baseline, the agent must simultaneously
learn two parameterized functions: the policy network $pi (a mid(bar) s; bold(theta))$
and the state-value network $hat(v)(s; bold(w))$. The value network weights
$bold(w)$ are typically updated using standard Monte Carlo evaluation (minimizing
the error between the prediction and $G_t$), while the policy weights
$bold(theta)$ are updated using the baseline-adjusted return:
$
    delta = G_t - hat(v)(S_t; bold(w)),\
    bold(theta)_(t+1) =
    bold(theta)_t + alpha_theta gamma^t delta nabla ln pi (A_t mid(bar) S_t; bold(theta)_t).
$

#figure(
    algo(
        title: [REINFORCE with Baseline],
        parameters: ([step sizes $alpha_theta > 0$, $alpha_w > 0$],),
        line-numbers: false,
    )[
        Initialize policy parameter $bold(theta)$ and state-value weights $bold(w)$\
        Loop for each episode:#i\
        Generate an episode $S_0, A_0, R_1, ..., S_(T-1), A_(T-1), R_T$ following $pi (dot mid(bar) dot; bold(theta))$\
        Loop for each step of the episode $t = 0, 1, ..., T - 1$:#i\
        $G <- sum_(k=t+1)^T gamma^(k-t-1) R_k$\
        $delta <- G - hat(v)(S_t; bold(w))$\
        $bold(w) <- bold(w) + alpha_w delta nabla hat(v)(S_t; bold(w))$\
        $bold(theta) <- bold(theta) + alpha_theta gamma^t delta nabla ln pi (A_t mid(bar) S_t; bold(theta))$#d\
        #d
    ],
    caption: flex-caption(
        [REINFORCE with Baseline],
        [REINFORCE with Baseline @Sutton2018],
    ),
    kind: "algo",
    supplement: "Algorithm",
) <alg:reinforce-baseline>

#heading([A Note on Encouraging Exploration], level: 3, numbering: none)

In policy gradient methods, the agent learns to increase the probability of
actions that yield high rewards. However, if the agent discovers a locally
optimal strategy early in training, it may rapidly increase the probabilities
of those specific actions to near certainty. When a policy becomes entirely
deterministic, exploration effectively halts, preventing the agent from ever
discovering the true global (or at least a better local) optimum.

To prevent this premature convergence, we can employ _entropy regularization_~@A3C.
In information theory, entropy measures the unpredictability of a random
variable. For a stochastic policy $pi$, the entropy at a given state $s$
is defined as:
$
    H (pi (dot mid(bar) s; bold(theta))) =
    - sum_(a in cal(A)(s)) pi (a mid(bar) s; bold(theta))
    ln pi (a mid(bar) s; bold(theta)).
$

By adding this entropy term to our objective function, weighted by a
temperature coefficient $beta > 0$, we actively reward the agent for maintaining
uncertainty in its action selection. The modified objective function becomes:
$
    J_"regularized" (bold(theta)) =
    J(bold(theta)) + beta EE_pi [H(pi (dot mid(bar) S_t; bold(theta)))].
$\

The regularization term forces the agent to balance two competing goals: maximizing
the expected return and keeping its action distribution as broad as possible.
$beta$ can also be decayed as training progresses, allowing the agent to
slowly transition from wide exploration to exploiting the optimal strategy.


= Implementing an Environment for Prší <chapter:environment>

With the theoretical methods established, this chapter
introduces the specific environment the agents will navigate.
First, we outline the rules of the card game Prší, followed by a discussion
on the architectural design of the #gls("rl") environment.

== The Rules of Prší

Prší~@prsi (translating to "it's raining") is a popular card game played in
the Czech Republic. It is played with a German-suited card deck,
which contains 32 cards -- from 7 to Ace across
four suits:
- Hearts #box(height: 0.95em, baseline: 20%, image("images/hearts.png")),
- Bells #box(height: 0.95em, baseline: 20%, image("images/bells.png")),
- Leaves #box(height: 0.95em, baseline: 20%, image("images/leaves.png")),
- Acorns #box(height: 0.95em, baseline: 20%, image("images/acorns.png")).
A full deck can be seen in @fig:bohemian-deck.
While not every household in Czechia abides by the same rules, the version
described below is the one we will be following during the implementation of
our environment.

The game is played with two to six players (unless playing with multiple decks).
Each player is dealt four cards
and one initial card is placed face-up onto the discard pile to start the game.
The remaining cards are placed face-down as the drawing pile. Players then
take turns clockwise, placing one of their cards onto the discard pile
until all players but one have no cards left.
To place a card on top of the discard pile, the card being placed
must match the top card in either suit or rank.
If a player has no playable cards, they must draw a card from the draw pile.
If there are no cards left in the draw pile, the discard pile gets flipped
over and becomes the draw pile (although the top card from the discard pile remains unflipped
for clarity).

In Prší, several special cards also have unique effects. Playing an Ace
forces the next player to skip a turn unless they also play an Ace. An Ober
(Queen equivalent) doesn't need to match either rank or suit of the top card
to be played, and the player playing an Ober can choose which suit the next played
card must belong to. Playing a 7 forces the next
player to draw two cards. If the next player instead also plays a 7, the player
after them is forced to draw four, etc. If a player has won (has no cards on hand),
they can be brought back during their winning round if the player before them plays
the 7 of hearts. They must then draw the relevant number of cards (two to eight)
and continue playing.

#figure(
    image("images/Bohemian_deck.png", width: 80%),
    caption: flex-caption(
        [Deck of Prší cards],
        [A deck of German cards (Bohemian pattern)~@bohemian-deck],
    ),
) <fig:bohemian-deck>

== Designing a Reinforcement Learning Environment

When developing #glspl("rl") agents for card games, frameworks like
RLCard~@rlcard-site can be used, as they provide a predefined
API for combining #gls("rl") with card games.
However, for this thesis, a custom environment
was implemented from scratch in Python @python.

This decision was driven by wanting to adhere to the modern Gymnasium API
(a maintained fork of OpenAI's Gym)~@gymnasium more closely.
Building the environment from the ground up also allowed for
complete control over the implementation of self-play.
Finally, our custom implementation utilizes Python type hints, which improves
code readability and maintainability.

To translate the rules of Prší into this programmable Gymnasium-like interface,
we must define the core components of the underlying #gls("mdp"):
the state observations, the action space, and the reward signal.

=== States and Observations

As discussed in @chapter:rl-intro, card games like Prší are inherently
#glspl("pomdp") because the true state of the game (the exact order of the
deck and the contents of the opponent's hand) is hidden from the players.

The environment internally tracks the full, perfect-information state
with the help of a `GameState` Python object, which includes
the top card, the currently active suit
#footnote([ Which can differ from the top card's suit due to an Ober being played.]),
and the active card effect (e.g., drawing penalty or skipped turn).
The environment object `PrsiEnv` itself also tracks the draw pile,
the discard pile, and the full hands of both players.

However, the agent only receives an observation containing the public
information and its own private information. Specifically, the agent observes
its own hand, the current top card, the active suit, the active effect, the
number of cards remaining in the opponent's hand, and whether the draw pile
has been flipped. By processing these observations, the agent must learn to
infer the hidden variables and strategize accordingly. How different
agents process the state will be discussed further in @chapter:experiments.

=== The Action Space

The action space defines the set of all possible moves an agent can make.
In Prší, a player can either draw a card from the deck or play a valid
card from their hand.

While the deck contains 32 cards, the action space must account for the special
mechanics of the Ober. When an Ober is played, the player
must also declare the active suit for the next turn. Therefore, playing an Ober
is not a single action, but four distinct actions (one for each suit).
The complete discrete action space consists of 45 possible actions:
- 1 action for drawing a card,
- 28 actions for playing a standard non-Ober card,
- 16 actions for playing an Ober and selecting one of the four suits.

#start-par Because an agent only holds a small subset of the deck
at any given time, and because the rules of the game restrict which
cards can be placed on the discard pile,
the vast majority of these 45 actions are illegal in any given
state. To handle this, the environment provides an action mask -- a boolean
array that flags valid actions. The agents use this mask to filter their
decision-making process, ensuring they only select legal moves. Were an agent
to ignore this mask and decide to cheat, the environment raises a `ValueError`.

=== Reward Signal and Termination

The goal of the agent is to win the game by emptying its hand. To reflect this,
the environment provides a sparse reward signal:
- A reward of $1$ is granted if the agent successfully plays its last card
    and wins the game.
- A reward of $-1$ is given if the opponent plays their last card.
- A reward of $0$ is returned for all non-terminal intermediate steps.

#start-par A single game represents one episode.
Because games of Prší can theoretically
enter infinite loops (e.g., players endlessly drawing and playing the same
sequence of cards), the environment enforces a truncation limit of 600 steps.
If the maximum step limit is reached, the game ends in a draw (a reward of~$0$)
to prevent the training loop from stalling.

Episodes are also terminated with a reward of $0$ in case the drawing pile
is emptied completely and a player (either the agent or the opponent) tries
to draw a card.

=== The Agent Interface

With the environment dynamics established, we define an abstract `Agent` class.
This ensures that whatever algorithm the agent relies on to choose
its actions, it exposes a consistent interface
for the environment to interact with. The interface can be seen
in @code:agent-interface.

#figure(
    ```python
    class Agent(ABC):
        def clone(self) -> "Agent":
            return deepcopy(self)

        @abstractmethod
        def choose_action(
            self, state: Any, hand: list[Card], info: dict[str, Any]
        ) -> Action:
            pass

        @abstractmethod
        def evaluate(self, *args, **kwargs) -> None:
            pass
    ```,
    caption: [Abstract Agent interface],
) <code:agent-interface>

The `choose_action` method is the core decision-making function, receiving a
state representation `state`, the agent's private `hand`,
and additional `info` (such as the opponent's card count).

Additionally, the `clone` method is a crucial
architectural component that enables self-play; it allows the environment
to instantiate a deep copy of the currently training agent to use
as the opponent for future episodes.

Two baseline agents using this interface are provided:
`RandomAgent`, which can either play a random card from its hand or draw a card,
and `GreedyAgent`, which also plays random cards, but only draws when no other
option is available.

=== Environment Dynamics and the Step Function

To finalize the programmable interface, our implementation models the game
as a 1-versus-1 environment. To make the environment compatible with standard
single-agent algorithms, we embed the opponent directly into the
environment's `step()` function. An extracted snippet is shown
in @code:env-step.

#figure(
    ```python
    def step(self, action: Action) -> tuple[GameState, float, bool, dict]:
        # ... (agent's chosen action gets executed) ...

        # If the game didn't end, the opponent immediately responds
        opponent_action = self._opponent.choose_action(
            self._state, self._opponent_player_info.hand, player_info
        )
        self._execute_action(self._opponent_player_info, opponent_action)

        # ... (return next state to the agent) ...
    ```,
    caption: [Environment step function],
) <code:env-step>

When the training agent takes an action, the environment executes it, immediately
queries the embedded opponent for its response, executes the opponent's action,
and only then returns the next state, a reward, a flag whether
the game has ended, and a dictionary `info` containing additional data
the agent may need (such as the cards in its hand).
From the perspective of the learning agent,
the environment simply transitions from one of its turns directly to its next turn,
abstracting away the multi-agent complexity.


= Experiments <chapter:experiments>

== Implementation of Agents

While the previous chapter introduces the `GameState` object that is
returned to the agent when it takes a step in the environment, as well
as an `info` dictionary, the information
contained in these is actually quite limited. There are things outside
these returned values that could be useful to the agent's decision-making
process.

A crucial aspect is that the agents might need memory of what cards were
played throughout the game. For example, it's a bad idea to end the game
with a heart-suited card if the 7 of hearts hasn't been played yet, as the other
player could return the winner into the game by playing that card. Several
implementations of memory were chosen.

=== Memory for Tabular Methods

Because the full state space of Prší is too large for the agents to remember
and represent directly,
we employ state abstraction to reduce the number of unique states.
For example, the `hand_state_option` allows the agent to see only
the count of cards in its hand (possibly bounded) rather than the full
set of cards. Similarly, the `played_subset` option allows to track
only a small number of critical cards that have already been played, such as
Sevens, Obers, and Aces.

This simplified representation allows the $Q$ table to remain small
enough to fit in memory while still capturing the most important aspects
of the game. By packing these abstracted features into a compact tuple
(alongside the values from the `GameState` object),
the tabular agents can learn efficiently
from millions of episodes with memory without exceeding
reasonable hardware limitations.

=== Memory for Deep Learning Methods

Unlike tabular methods, which require discrete states to use as dictionary keys,
deep learning algorithms like #gls("dqn") and REINFORCE require the environment
state to be represented as a continuous, fixed-size numerical tensor.
Furthermore, to ensure stable gradient updates during backpropagation,
these inputs must be carefully scaled.

To achieve this, the `GameState` and the history of played cards are flattened
into a 1D vector of 32-bit floats. We utilize two primary encoding strategies:
_normalization_, which scales integer counts to a range between 0 and 1
(e.g., dividing the opponent's card count by a theoretical maximum of 31),
and _one-hot encoding_, which represents categorical variables like suits
or specific cards as a sparse binary array.

The complete observation vector for the neural networks consists of:
- _Hand Representation:_ Configurable as either a normalized card count
    or a full 32-element one-hot encoded array representing exact held cards.
- _Opponent Card Count:_ A single normalized float.
- _Top Card:_ A 32-element one-hot encoded array.
- _Active Suit:_ A 4-element one-hot encoded array.
- _Card Effect:_ A 3-element one-hot encoded array (no effect, drawing a card,
    skipping a turn).
- _Effect Strength:_ When drawing cards because of 7s, signals how many to draw.
- _Played Cards Memory:_ An array tracking played cards (e.g., just Sevens,
    special cards, or the full deck), normalized by the maximum possible
    count of 4 for each rank.

#start-par Because neural networks naturally generalize across similar inputs,
they do not
suffer from the same state space explosion as tabular methods. This allows
our deep #gls("rl") agents to train on the "full" hand representation and the
complete 32-card memory tracking without exhausting system resources.
Using abstracted state spaces from the tabular methods is
still available, however.

The architectures used for these agents are standard feed-forward
#glspl("mlp") with the hidden layers being configurable by both depth
and width (although width is shared between layers). The implementation
utilizes the Adam optimizer~@Adam.

=== Limitations of Memory Representation

One problem the agents don't solve is that if a deck is flipped over once
the draw pile is exhausted, there is an incredible amount of
information available, as the order of cards in the draw pile is now
known and deterministic. When representing memory as a set, the order of cards
is lost and therefore the agents can't use it. Once a deck is flipped
(information included in the `info` dictionary), the memory buffer is simply
reset and the agent starts "from scratch".

== Agent Evaluation

To determine the effectiveness of the implemented #gls("rl") algorithms, each
agent was subjected to a continuous training phase followed by a strict,
reproducible evaluation. For all agents, we used a discount factor
$gamma = 0.99$.

Due to hardware constraints, each algorithm was allowed to train for a maximum
of 24 hours. Because tabular methods and deep neural networks process
environment steps at vastly different computational speeds, the total number of
episodes completed within this timeframe varied significantly between the
different approaches. To monitor the learning progress during this 24-hour
window, the agents continuously logged their _batch win rate_ -- the percentage
of games won over the most recent "logging batch" of training episodes.

Following the conclusion of the training phase, the saved models were
evaluated. The evaluation protocol consisted of $1000$ independent games
against the `GreedyAgent` baseline. To ensure fair and reproducible comparisons
between different hyperparameters, these evaluation games were run using a fixed
random seed. Furthermore, any exploratory behaviour used during training
(such as $epsilon$-greedy random actions)
was strictly disabled. This allowed us to measure the true, underlying strength
of the agent's deterministic strategy.

The configuration that achieved the highest win rate against the `GreedyAgent`
for each algorithm was crowned the best model for that approach. Those
models were subsequently evaluated against the `RandomAgent`
to ensure its strategy was robust and capable of generalizing against
a different opponent.

=== Monte Carlo

The Monte Carlo agents were trained using the heavily abstracted state space to
ensure the $Q$ table could fit into memory. Across all runs, the hand size was
truncated to a maximum of 4, and the memory of played cards was restricted to
the `specials` subset (Sevens, Obers, and Aces).

We experimented with several hyperparameters, comparing first-visit against
every-visit updates, constant exploration rates ($epsilon = 0.1$ and
$epsilon = 0.05$) against a decaying exploration rate, and a simple incremental
mean update against an exponential moving average with a fixed step size ($alpha$).
The learning curves over the training period can be seen in @fig:mc-training,
and the final evaluation results are summarized in @tab:mc-results.

#figure(
    image("images/monte_carlo_training.svg", width: 100%),
    caption: flex-caption(
        [Monte Carlo Training Curves],
        [Batch win rates of various Monte Carlo configurations during training],
    ),
) <fig:mc-training>

#figure(
    table(
        columns: 2,
        align: (left, center),
        [ *Configuration* ], [ *Win Rate* ],
        [ First-visit, $epsilon=0.1$ ], [ 49.80% ],
        [ Every-visit, $epsilon=0.1$ ], [ 48.40% ],
        [ First-visit, $epsilon=0.05$ ], [ 47.70% ],
        [ Decaying $epsilon$, $alpha=0.01$ ], [ 45.20% ],
        [ Decaying $epsilon$, $alpha=0.05$ ], [ 16.10% ],
        [ Decaying $epsilon$, $alpha=0.1$ ], [ 9.00% ],
        [ Decaying $epsilon$, Self-play, $alpha=0.1$ ], [ 6.80% ],
    ),
    caption: [Monte Carlo evaluation win rates against `GreedyAgent`],
) <tab:mc-results>

The most successful configuration utilized a simple incremental mean for its
updates and a constant $epsilon = 0.1$, achieving a win rate of 49.80% against
the `GreedyAgent`. Interestingly, first-visit updates slightly outperformed
every-visit updates in this environment. A difference of 1.2% (12 games) could
just be random noise, however.

A drop in performance occurred when introducing a fixed step size $alpha$.
As $alpha$ increased from 0.01 to 0.1, the agent's win rate plummeted to 9.00%.
This indicates that weighing recent episodes too heavily aggressively destabilized
the $Q$ value estimates. Furthermore, training via self-play completely collapsed
(6.80% win rate).

While the best Monte Carlo agent effectively tied with the `GreedyAgent`
(falling just short of a $>50%$ win rate), it dominated when
evaluating this best configuration against the `RandomAgent`.
It achieved a 92.00% win rate, proving that it successfully learned
a strategy capable of generalizing against different opponents.

=== Q-Learning

To ensure a fair comparison with the Monte Carlo approach, the tabular Q-Learning
agents were trained using the exact same abstracted state space configuration: a
maximum truncated hand size of 4 and a memory of played cards restricted to the
`specials` subset.

Because Q-Learning is a #gls("td") method that updates its estimates incrementally
step-by-step, it inherently requires a fixed step size $alpha$. Therefore, our
hyperparameter search focused on the interplay between the learning rate $alpha$
and the exploration strategy (constant $epsilon$ versus decaying $epsilon$), as
well as the impact of self-play. The learning curves for these runs can be seen in
@fig:ql-training, with final evaluation results summarized in @tab:ql-results.

#figure(
    image("images/q-learning_training.svg", width: 100%),
    caption: flex-caption(
        [Q-Learning Training Curves],
        [Batch win rates of various Q-Learning configurations during training],
    ),
) <fig:ql-training>

#figure(
    table(
        columns: 2,
        align: (left, center),
        [ *Configuration* ], [ *Win Rate* ],
        [ Constant $epsilon=0.1$, $alpha=0.1$ ], [ 40.10% ],
        [ Decaying $epsilon$, $alpha=0.01$ ], [ 34.80% ],
        [ Constant $epsilon=0.05$, $alpha=0.1$ ], [ 33.50% ],
        [ Decaying $epsilon$, $alpha=0.05$ ], [ 16.60% ],
        [ Decaying $epsilon$, $alpha=0.1$ ], [ 15.10% ],
        [ Decaying $epsilon$, Self-play, $alpha=0.1$ ], [ 15.00% ],
        [ Decaying $epsilon$, $alpha=0.2$ ], [ 13.10% ],
    ),
    caption: [Q-Learning evaluation win rates against `GreedyAgent`],
) <tab:ql-results>

The highest-performing Q-Learning configuration utilized a constant exploration
rate of $epsilon = 0.1$ and a step size of $alpha = 0.1$, achieving a 40.10% win
rate against the `GreedyAgent`. Similar to the Monte Carlo results, constant
exploration proved to be more effective than a decaying exploration schedule.

When observing the runs utilizing a decaying $epsilon$, it becomes clear that
Q-Learning is also sensitive to the step size parameter. A relatively small
step size ($alpha = 0.01$) allowed the agent to learn a passable strategy
(34.80% win rate), but increasing $alpha$ beyond $0.05$ caused the performance
to collapse entirely. This could suggest that in the highly stochastic environment of
a card game, bootstrapping off of newly observed, noisy transitions with a high
learning rate prevents the $Q$ values from converging cleanly. Additionally,
the self-play paradigm failed to yield positive results here as well, mirroring
the collapse seen in the Monte Carlo agent.

Overall, the best Q-Learning agent noticeably underperformed the best Monte Carlo
agent against the `GreedyAgent` (40.10% versus 49.80%). While the cause
of this discrepancy is not definitively known, a likely culprit could be
the maximization bias, causing overestimations in certain states.

However, despite struggling against the greedy baseline, the best Q-Learning
agent still defeated the `RandomAgent` with an 88.00% win rate. This
shows that the algorithm somewhat successfully mapped the abstracted state space
and learned a generally usable policy, even if its ultimate ceiling was lower
than its Monte Carlo counterpart.

=== (Double) Deep Q-Network

Unlike the tabular methods, the #gls("dqn") and #gls("ddqn") agents
were trained using the full, unabstracted state representation. This utilized
a 32-element one-hot array for the agent's hand and a complete
32-element array tracking all played cards. Because neural networks
can naturally generalize across continuous inputs, this allowed the agents to
observe the exact state of the game without relying on manual feature engineering.
One might hope that this full representation will allow these agents to learn
a much stronger estimate. This was, however, not the case.

Training these deep architectures is computationally expensive. Within the 24-hour
hardware limit, these agents experienced significantly fewer environmental steps
than the tabular methods. We experimented with a learning rate of $5 times 10^(-5)$,
a target network update frequency of 100 steps, and a batch size of 32. Our primary
variable of interest was network capacity, comparing a 2-layer network against a
deeper 4-layer network (both with 1024 neurons per hidden layer). The final evaluation
results for both algorithms are summarized in @tab:dqn-results.

#figure(
    image("images/dqn_training.svg", width: 100%),
    caption: flex-caption(
        [Deep Q-Network Training Curves],
        [Batch win rates of DQN configurations during training],
    ),
) <fig:dqn-training>

#figure(
    image("images/double_dqn_training.svg", width: 100%),
    caption: flex-caption(
        [Double Deep Q-Network Training Curves],
        [Batch win rates of DDQN configurations during training],
    ),
) <fig:ddqn-training>

#figure(
    table(
        columns: 3,
        align: (left, left, center),
        [ *Algorithm* ], [ *Configuration* ], [ *Win Rate* ],
        [ DQN ], [ 4 layers, $epsilon=0.05$ ], [ 27.60% ],
        [ DDQN ], [ 4 layers, $epsilon=0.05$ ], [ 25.00% ],
        [ DDQN ], [ 4 layers, $epsilon=0.1$ ], [ 20.20% ],
        [ DDQN ], [ 2 layers, $epsilon=0.1$ ], [ 0.30% ],
        [ DQN ], [ 2 layers, $epsilon=0.05$ ], [ 0.00% ],
    ),
    caption: [DQN and DDQN evaluation win rates against `GreedyAgent`],
) <tab:dqn-results>

The most immediate takeaway from these results is the stark reliance on network
capacity. Across both standard #gls("dqn") and #gls("ddqn"),
the shallower 2-layer architectures
completely collapsed, returning win rates of essentially zero. This indicates that
mapping the complex combinations of a full 32-card one-hot state representation
to accurate $Q$ values probably requires a highly non-linear function that a
2-layer network with 1024 neurons in each layer cannot adequately
approximate in this environment.

However, even with 4 hidden layers, the performance of the deep value-based
methods was poor compared to the tabular approaches. The best #gls("dqn") model
achieved only a 27.60% win rate against the `GreedyAgent`,
and its #gls("ddqn") counterpart achieved 25.00%.

Interestingly, standard #gls("dqn") slightly outperformed #gls("ddqn"). While
#gls("ddqn") is explicitly
designed to prevent the overestimation of action values, in environments with
highly sparse and delayed rewards like Prší, a slight overestimation bias could
possibly have acted as a form of optimistic exploration. It is, however, entirely
possible that this was just noise and "bad luck" on the #gls("ddqn") part.

When evaluated against the entirely random baseline (`RandomAgent`),
the best #gls("dqn") model secured a 53.40% win rate (and #gls("ddqn") secured
47.20%). This shows that the network learned essentially just random noise,
likely not even being able to learn that drawing when unnecessary is generally
a sub-optimal action (which is presumably what makes even the tabular methods
capable of defeating `RandomAgent`).
It heavily underperformed the tabular Monte Carlo agent's 92.00% win rate.
In a sense, the model that loses almost all the games is actually more
interesting, as it essentially means the agent somehow learned that drawing
cards is the best action.

=== REINFORCE

The final algorithm evaluated was the REINFORCE policy gradient method. Like the
#gls("dqn") agents, it utilized the full, unabstracted state representation.
However, while the value-based #gls("dqn") agents
failed to learn using a 2-layer network and achieved subpar results
with a 4-layer network, the REINFORCE agents successfully
utilized even the shallower architecture of 2 hidden layers with 1024 neurons each.

Because policy gradient methods learn a parameterized probability distribution over
actions rather than deterministic value estimates, our hyperparameter tuning focused
on variance reduction and exploration. Specifically, we tested the impact of a
learned state-value baseline, various coefficients for entropy
regularization ($beta$), and training via self-play. The learning curves are shown
in @fig:reinforce-training, and the final evaluation metrics are summarized in
@tab:reinforce-results.

#figure(
    image("images/reinforce_training.svg", width: 100%),
    caption: flex-caption(
        [REINFORCE Training Curves],
        [Batch win rates of various REINFORCE configurations during training],
    ),
) <fig:reinforce-training>

#figure(
    table(
        columns: 2,
        align: (left, center),
        [ *Configuration* ], [ *Win Rate* ],
        [ Baseline, $beta=0.05$ ], [ 64.90% ],
        [ Baseline, $beta=0.01$ ], [ 63.60% ],
        [ Baseline, $beta=0.05$, Self-play ], [ 62.40% ],
        [ No Baseline, $beta=0.01$ ], [ 62.10% ],
        [ Baseline, $beta=0.1$ ], [ 57.40% ],
        [ Baseline, $beta=0.001$ ], [ 55.10% ],
    ),
    caption: [REINFORCE evaluation win rates against `GreedyAgent`],
) <tab:reinforce-results>

The results mark a significant breakthrough in performance. The optimal REINFORCE
configuration (utilizing a baseline and an entropy coefficient of $beta=0.05$)
achieved a highly decisive 64.90% win rate against the `GreedyAgent`. This is the
only algorithm that confidently and consistently defeated the greedy baseline,
proving that it discovered a superior, long-term strategic policy. Against the
`RandomAgent`, this model reached a near-perfect 95.70% win rate.

The data illustrates the delicate balance required when applying entropy
regularization. A coefficient that is too low ($beta=0.001$) caused the policy to
prematurely converge on sub-optimal strategies, dropping the win rate to 55.10%.
Conversely, a coefficient that is too high ($beta=0.1$) forced the agent to behave
too randomly, degrading performance to 57.40%. The "Goldilocks zone" was found at
$0.05$, which provided just enough exploration to map the complex state space
without diluting the final deterministic policy.

While the baseline-equipped agent marginally outperformed the standard REINFORCE
agent (63.60% versus 62.10% at $beta=0.01$), this slight improvement is likely
just statistical noise. Because the environment's reward signal is strictly sparse
and bounded between $-1$ and $1$, the true expected value of most non-terminal
states probably hovers near $0$. Consequently, subtracting this near-zero baseline
from the return provides minimal variance reduction in practice.

However, it is
highly noteworthy that the baseline network did not diverge. The baseline relies
on a continuous state-value function approximation trained via Monte Carlo returns.
The fact that this value network remained stable stands in stark contrast to the
#gls("dqn") results, where similar deep value-approximation architectures
completely collapsed. This suggests that estimating deep value functions
via Monte Carlo returns is more stable in this
highly stochastic environment than relying on temporal difference bootstrapping.
This is further supported by the fact that tabular #gls("mc") methods outshined
Q-Learning during their evaluation against `GreedyAgent`.

Finally, unlike the value-based algorithms, REINFORCE successfully maintained its
stability during self-play. The self-play configuration secured a fairly
competitive 62.40% win rate. While notable, this is still lower than what the
agents trained against the baseline achieved and somewhat follows the trend from
the value-based methods.

=== Selecting the Best Agent

To select the best agent, which will be tested against
real human opponents, the single best-performing
configuration from each algorithm family was compared side-by-side. The summary
of their win rates against both the `GreedyAgent` and `RandomAgent` baselines is
presented in @tab:best-agents.

#figure(
    table(
        columns: 3,
        align: (left, center, center),
        [ *Algorithm* ], [ *Win Rate vs. Greedy* ], [ *Win Rate vs. Random* ],
        [ REINFORCE ], [ 64.90% ], [ 95.70% ],
        [ Monte Carlo ], [ 49.80% ], [ 92.00% ],
        [ Q-Learning ], [ 40.10% ], [ 88.00% ],
        [ (D)DQN ], [ 27.60% ], [ 53.40% ],
    ),
    caption: [Performance summary of the best agents from each algorithm family],
) <tab:best-agents>

Looking at the aggregated results, a clear hierarchy emerges. The deep policy
gradient method, REINFORCE, stands out as the champion of this
environment. It was the only algorithm to conclusively defeat the
greedy baseline and achieved near-perfect consistency against random play.

The tabular methods performed respectably well given their heavily abstracted
state space constraints. Monte Carlo, in particular, proved to be a highly
stable approach, effectively fighting the greedy baseline to a standstill.
However, their reliance on manual feature abstraction ultimately placed a ceiling
on their strategic capabilities, which can be seen in their respective graphs.
Conversely, the deep value-based methods
struggled heavily to parse the full state representation within the time constraints,
falling far behind even the simplest tabular approaches.

Because of its dominant performance, the REINFORCE agent (configured with a
state-value baseline and an entropy coefficient of $beta=0.05$) was officially
selected as the final representative model. This specific agent represents the
culmination of the training efforts and serves as the primary AI opponent
for the human evaluation phase detailed in the following section.

== Performance Against Human Players

=== Command-Line Interface
To evaluate the final trained agent against real human players, a custom
`HumanAgent` was implemented. Unlike the computational baselines or trained
neural networks, this agent acts as an interactive command-line
interface wrapper.

During its `choose_action` call, the agent pauses the environment loop
and visually renders the public game state directly in the terminal.
This includes the current top card, the active suit (utilizing Unicode
icons for visual clarity), the number of cards remaining in the opponent's hand,
and the human player's private hand. The human player then inputs their
decision via standard input. The agent validates this input, prompting for
additional information if an Ober is played to select the next suit,
and translates the final decision back into the environment's internal
action format.

To facilitate evaluation over multiple independent play sessions,
the agent features persistent statistical tracking. The results of each
game are appended to a JSON file, recording the cumulative wins,
total games played, and the overall win rate of the human tester.

=== Results of Human Evaluation

To establish a baseline for human performance, an initial testing phase was
conducted where a single human player competed against the `GreedyAgent`
for a total of #human_greedy_games games. In this baseline evaluation,
the human player secured #human_greedy_wins victories, resulting in a
#human_greedy_rate% win rate. This confirms that while the greedy strategy is
competent, a human player can consistently exploit its predictable,
short-sighted nature. It also indirectly puts the REINFORCE agent
on par with human players, in the sense that they have achieved
a similar win-rate (although #human_greedy_games games is a relatively small
sample size).

Next, the human testers faced the champion REINFORCE agent. Over a series of
#human_rl_games games, human players achieved #human_rl_wins victories, yielding
a win rate of #human_rl_rate%.

Comparing these two results reveals the tangible strength of the learned policy.
The REINFORCE agent successfully lowered the human win rate by several percentage
points compared to the greedy baseline. While the human players still maintained a
positive win record overall -- highlighting the inherent difficulty of achieving
superhuman performance in imperfect-information games without look-ahead planning
algorithms -- the REINFORCE agent proved to be a noticeably more difficult and
resilient adversary.


= Discussion and Future Work

The experiments conducted in this thesis demonstrate that #gls("rl")
can be applied to the imperfect-information environment of Prší with a
reasonable degree of success.
While tabular methods achieved competent play through heavy state abstraction,
the deep policy gradient algorithm, REINFORCE, proved capable of mapping the
full state representation to a competitive strategy. However, the evaluation
also highlighted several limitations in the current approach and environment
implementation, opening several avenues for future research.

== Computational Limitations and Language Choice

One of the most significant bottlenecks during the training phase was computational
efficiency. The custom environment and training loops were implemented entirely in
Python. While Python is the standard for machine learning research due to its
rich ecosystem, its interpreted nature makes
continuous environment interactions inherently slow.

Within the strict 24-hour training limit, this slow execution speed heavily
penalized the deep learning models. The tabular methods processed millions of
episodes, whereas the neural networks processed far fewer. It is possible
that the failure of the deep value-based methods (#gls("dqn") and #gls("ddqn"))
was not due to algorithmic incompatibility, but simply a lack of sufficient
experience to converge. Rewriting the core `PrsiEnv` and the simulation loop in a
compiled language or utilizing a Just-In-Time compiler like JAX
could increase the simulation throughput, allowing deep
architectures to train much more effectively. The main benefit of the selected
approach was the code readability provided by using
industry-standard technologies.

== State Representation, Architectures, Hyperparameters

The current deep learning agents utilized a flattened, 1D one-hot encoded vector
to represent the game state. For #gls("dqn") in particular, this sparse representation
likely contributed to the network's inability to learn an accurate value function.
This thesis did not explore different state representations for the value-based
agents, such as passing the abstracted state space used by the tabular agents
into the neural networks.
This could make the #gls("dl") approaches converge to at least a similar level
as the tabular methods. Overall, there are many hyperparameter combinations
that were not tried due to time and hardware constraints that could result
in #gls("dqn") convergence.

Furthermore, dealing with imperfect information currently relies on a fixed
"played cards" memory array, effectively representing a set, an unordered
container.
As noted in @chapter:experiments, this static representation completely
loses the sequential order of cards, which becomes critical information once the
draw pile is exhausted and the discard pile is flipped. To achieve "real" memory
that can dynamically track the flow of the game and infer the opponent's hand
probabilities, the feed-forward #glspl("mlp") could take a learned representation
of a state embedded by a #gls("rnn") layer using for example
#gls("lstm")~@LTSM cells. This would allow the agent to maintain a hidden state
vector that updates with every played card, naturally resolving
the #gls("pomdp") nature of the game without relying on handcrafted memory arrays.

== Advanced Reinforcement Learning Algorithms

While REINFORCE proved successful, it is one of the most fundamental policy gradient
methods and suffers from high sample inefficiency. Upgrading the agent to use
state-of-the-art actor-critic algorithms, such as #gls("ppo")~@PPO,
could improve training stability and allow for multiple
epochs of learning on the same batch of data.

Furthermore, the naive self-play mechanism implemented in this thesis yielded
sub-optimal results, causing the tabular agents to collapse and also
degrading the performance of REINFORCE.
To resolve these instabilities and look beyond standard model-free algorithms,
MuZero~@MuZero represents the frontier of board game AI. MuZero builds
a predictive model of the environment's dynamics and uses #gls("mcts")
to plan ahead, naturally integrating highly robust self-play
training loops. Adapting MuZero for the stochastic, hidden-information
environment of Prší could not only stabilize self-play dynamics, but also
bridge the gap between the current agent's reactive playstyle and the proactive,
look-ahead strategies employed by skilled human players.

== The Environment

Currently, the training environment is strictly limited to 1-versus-1 interactions.
However, Prší is traditionally played as a multiplayer game (typically 3 to 6 players).
Expanding the `PrsiEnv` to support an arbitrary number of players would introduce
new strategic layers. This option was left out to limit the scope of our
experiments, but remains an interesting future possibility.

Another modification that could prove interesting is reward shaping. A flat
$-1$ on a loss and $+1$ on a victory was chosen to ensure the agents find their own
way to win without influencing their strategy in any way. It could however
be interesting to see how different rewards would affect the tested methods.
For example -0.01 for drawing a card, or even entirely leaving out wins and
losses, with purely using the negative number of cards in hand as the
reward in each timestep.

#heading([Conclusion], numbering: none)

The primary goal of this thesis was to evaluate the application of modern
#gls("rl") algorithms in the hidden-information environment of the
Czech card game Prší, and to train an artificial agent capable of challenging
human opponents.

To achieve this, a custom training environment was implemented
from scratch in Python, mostly conforming to the standard Gymnasium interface.
We used the environment to evaluate two distinct
classes of #gls("rl") algorithms:
tabular value-based methods (Monte Carlo and Q-Learning) and deep learning
approaches (#gls("dqn") and REINFORCE). Due to the exponential size of the true
state space, the tabular agents were restricted to heavily abstracted state
representations. Despite this handicap, the tabular Monte Carlo approach proved
highly resilient, nearly matching the performance of a greedy baseline agent.
The deep value-based #gls("dqn") agents unfortunately failed to stabilize
with their selected hyperparameters within the training constraints
when exposed to the full, unabstracted game state.

The most significant achievement was the success of the REINFORCE algorithm.
It successfully mapped the full one-hot encoded state space into a
reasonable policy. It was the only algorithm to conclusively and consistently
defeat the `GreedyAgent` baseline, achieving a win rate of nearly 65%.

In the final evaluation phase, the most successful REINFORCE agent was deployed
against human players via a simple command-line interface. Over a sample of
#human_rl_games games, the human players achieved a win rate of #human_rl_rate%.
While the humans ultimately won more often than not,
this win rate represents a notable drop in
human performance compared to games against the baseline heuristic.

This thesis demonstrates that while deep #gls("rl") can
be highly sensitive to hyperparameter selection,
policy gradient methods can learn non-trivial strategies
in stochastic card games without any prior human knowledge. The developed
environment provides a foundation for future research, particularly
in exploring #gls("rnn")-based state representations or more advanced #gls("rl")
methods like #gls("ppo") or MuZero.


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
