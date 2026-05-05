#import "./template/template.typ": *
#import "@preview/glossarium:0.5.10": Gls, Glspl, gls, glspl, make-glossary, print-glossary, register-glossary
#import "@preview/algo:0.3.6": algo, code, comment, d, i
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

// Unless otherwise noted, the mathematical notation, foundational definitions,
// and general algorithmic frameworks presented in this chapter closely follow
// the conventions established in @Sutton2018.

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
        [Markov Decision Process @npfl139-lec01],
    ),
) <fig:mdp-loop>

To formalize any reinforcement learning problem, we model the environment as
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
To model environments like these, we define the #gls("pomdp")~@Spaan2012.

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
illustration can be seen in @fig:pomdp-loop.~@Sutton2018 @npfl139-lec01

#figure(
    image("images/pomdp.png", width: 80%),
    caption: flex-caption(
        [Partially Observable Markov Decision Process],
        [Partially Observable Markov Decision Process @npfl139-lec01],
    ),
) <fig:pomdp-loop>


=== Reward vs. Return

While we have said that the goal of the agent is to maximize the reward it gets,
it wasn't an entirely accurate formulation.
The goal of the agent is to maximize the _cumulative_
reward over the whole interaction.

Let's take
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
for episodic tasks).~@Sutton2018 @npfl139-lec01

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
value of the expected next state. This is known as the *Bellman equation*
for $v_pi$:
$
    v_pi (s) = EE[G_t mid(bar) S_t = s]
    = sum_a pi(a mid(bar) s) sum_(s', r) p(s', r mid(bar) s, a) [r + gamma v_pi (s')]
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

#Gls("mc") methods are a fundamental class of reinforcement learning algorithms
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
*First-visit #gls("mc")* averages only the returns following the first time a
state-action pair is visited in an episode, whereas *every-visit #gls("mc")*
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
settings, this is often handled by the assumption of *exploring starts*, where
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
in @alg:mc-control.~@Sutton2018 @npfl139-lec01

#figure(
    algo(
        title: [Monte Carlo Control],
        parameters: ([episodes], $epsilon$, $gamma$),
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
        [First-visit Monte Carlo Control with $epsilon$-greedy exploration @Sutton2018],
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

Q-Learning elegantly bypasses this issue by utilizing #gls("td") learning.
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
        [Q-Learning: Off-policy TD control algorithm @Sutton2018],
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

#[
    // The text after the equation is only one line,
    // this prevents widowing
    #show math.equation.where(block: true): set block(sticky: true)

    $
        hat(v)(s; bold(w))\
        hat(q)(s, a; bold(w)).
    $
    In the field of #gls("rl"), this approach is fittingly
    called _function approximation_.
]

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
distribution $mu(s)$ and the true values $q_pi (s, a)$ are unknown,
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
a separate *target network* and *experience replay*.

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
To solve this, #gls("dqn") introduces a separate *target network*
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
        [Deep Q-Learning with experience replay and target networks @dqn2015],
    ),
    kind: "algo",
    supplement: "Algorithm",
) <alg:dqn>

== Policy Gradient Methods <chapter:policy-methods>

=== REINFORCE

#lorem(70)

#lorem(100)


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


= Discussion and Future Work // TODO: maybe future work should be a subsection

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
