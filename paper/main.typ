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
However, an area that remains much less explored are games where
a massive part of the world-state remains unknown to the player.

In this thesis, we will focus on Prší, a popular card game in the Czech Republic.
It is one of the many variants of the German game Mau-Mau and is not dissimilar to the world-famous card game Uno.
As a card game where everything but your hand and the cards already played
is a mystery to the agent, it provides a suitable environment for exploring
#gls("rl") methods in a unique setting.


#heading([Goals], depth: 2, numbering: none, outlined: true)

The main goal of this thesis is to evaluate different #gls("rl") algorithms on their
performance in Prší.

To achieve this, we will implement an environment capable of putting various agents against
each other in a 1v1 setting with 2 baseline opponents available. The environment will also be able to update the opponent
on the fly for self-play compatibility.

We will use the implemented environment to train agents using both tabular methods, such as Monte Carlo,
and #gls("dl") based approaches, like #gls("dqn", first: false).

Finally, after comparing the agents to a baseline to find the best one, we will test
its performance against human players and discuss the results.


= Reinforcement Learning <chapter:rl>

In this chapter, we introduce #gls("rl") and some of its formalisms and key concepts
in @chapter:rl-intro -- such as #glspl("mdp", first: false) and value functions.
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

The general idea of the environment is that the agent finds himself in some state,
from which the agent takes an action. The action will result in a reward and
move the agent into a different state, where the loop begins anew.
Formally, we'll model these environments as #glspl("mdp", display: "Markov Decision Processes").

=== Markov Decision Process

To formalize any reinforcement learning problems, we model the environment as
a #gls("mdp", first: true). This model captures the interaction between
an agent and its environment through states, actions, transition probabilities,
and rewards. It is widely used because it allows learning algorithms to reason
about long-term consequences of decisions rather than only immediate outcomes.
An illustration of an #gls("mdp") can be seen in @fig:mdp-loop.

// A formal paragraph introducing dynamics p, states, actions, rewards, etc.
// should go here. Cite either @rl-an-introduction or @npfl139-lec01

// Markov property of state - independence on the past, from intro to RL book

#figure(
    image("images/mdp.png", width: 80%),
    caption: flex-caption(
        [MDP illustration],
        [MDP illustration @npfl139-lec01],
    ),
) <fig:mdp-loop>

#lorem(50)

#figure(
    image("images/mdp.png", width: 80%),
    caption: flex-caption(
        [POMDP illustration],
        [POMDP illustration @npfl139-lec01],
    ),
) <fig:pomdp-loop>

#lorem(50)

=== (Action-)Value function

== Value-Based Methods <chapter:value-methods>

#lorem(40)

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


#heading([Future work], numbering: none)

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


= An example appendix

#lorem(100)

$
    sum_(i=1)^(infinity) 1 / i
$

```cpp
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}
```
