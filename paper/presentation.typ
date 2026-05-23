#let fit-blue = rgb("0064bb")
#let logo = "template/res/logo-fit-en-blue.svg"
#let title = [Reinforcement Learning for Prší Card Game]
#let author = [Josef Vašata]

#set page(
    paper: "presentation-4-3",
    margin: (x: 2cm, top: 2.5cm, bottom: 2cm),
    header: context {
        if counter(page).get().first() > 1 {
            set align(bottom)
            grid(
                columns: (1fr, auto),
                align: horizon,
                [
                    #v(0.3em)
                    #text(size: 12pt, fill: gray, title)
                ],
                image(logo, height: 1.5em),
            )
            v(-0.6em)
            line(length: 100%, stroke: 0.5pt + gray)
        }
    },
    footer: context {
        if counter(page).get().first() > 1 {
            line(length: 100%, stroke: 0.5pt + gray)
            v(-0.4em)
            grid(
                columns: (1fr, auto),
                text(size: 12pt, fill: gray, author),
                text(size: 12pt, fill: gray, counter(page).display()),
            )
        }
    },
)

#set text(font: "DejaVu Sans", size: 22pt, lang: "cs")
#set list(indent: 1em)

#let slide(title, body) = {
    pagebreak(weak: true)
    v(0.5em)
    block(
        width: 100%,
        inset: (bottom: 0.5em),
        text(size: 32pt, weight: "bold", fill: fit-blue, title),
    )
    v(1em)
    body
}

#page(header: none, footer: none)[
    #align(center + horizon)[
        #image(logo, width: 25%)
        #v(1em)
        #text(size: 34pt, weight: "bold", fill: fit-blue)[#title]
        #v(0.5em)
        #text(size: 24pt)[#author]
        #v(3em)
        #grid(
            columns: (1fr, 1fr),
            align(left)[
                #text(size: 15pt)[*Supervisor:* Ing. Daniel Vašata, Ph.D.]
            ],
            align(right)[
                #text(size: 15pt)[#datetime.today().display("[month repr:long] [year]")]
            ],
        )
        #v(2em)
    ]
]

#slide("Motivation")[
    - Prší -- popular card game in the Czech Republic
    - Interesting problem for AI
        - High level of stochasticity
        - Incomplete information
]

#slide("Goals")[
    1. Implement an environment for Prší
    2. Compare multiple RL approaches:
        - Tabular methods
        - Deep learning methods
    3. Evaluation:
        - Win-rate against greedy baseline
        - Win-rate against humans
]

#slide("Formalizing The Problem")[
    - The game is modeled as a _partially observeable Markov decision process_ (POMDP)
    #align(center, image("images/pomdp.png", height: 50%))
]
// TODO: Introduce return and Q function

#slide("Algorithms")[
    - Tabular methods -- Monte Carlo, Q-Learning:
        - Based on estimating the Q function
        - Descrete state space
    - Deep Q-Network (DQN):
        - Metoda založená na odhadu Q funkce
        - Uses neural networks
        - Similar states $=>$ similar Q estimate
    - REINFORCE:
        - *Policy gradient* method (Directly optimize the policy
            $pi_bold(theta) (a mid(bar) s)$)
]

#slide("Training Results")[
    #grid(
        columns: (1.2fr, 1fr),
        gutter: 1em,
        [
            - *REINFORCE* achieved *65% win-rate* against the greedy agent
            - Tabular methods reached almost 50%
            - DQN diverged (win-rate under 30%)
        ],
        [
            #image("images/reinforce_training.svg", width: 100%)
            #align(center)[#text(size: 14pt)[_Průběh úspěšnosti REINFORCE během trénování_]]
        ],
    )
]

#slide("Evaluation Against Human Players")[
    - Performance comparison
        - Human vs. greedy agent: ~65% human win-rate
        - Human vs. REINFORCE: *~54%* human win-rate
    - The REINFORCE agent lowered the human win-rate significantly,
        performance was almost even
]

// TODO: add human agent CLI screenshots

#slide("Value of The Work")[
    - Environment implementation:
        - Extensible Prší environment written in Pythonu\ (Gymnasium API, self-play support)
    - Evaluation against human players:
        - A custom CLI was created to compare agent performance to humans
]

#slide("Conclusion and Future work")[
    - Policy gradient methods managed to learn a non-trivial policy
      even through the game's high level of stochasticity
    - Future work:
        - State representation: Using RNNs
        - Advanced algorithms: Proximal Policy Optimization (PPO),
            Soft Actor-Critic (SAC), MuZero
]

#slide("")[
    #align(center + horizon)[
        #text(size: 40pt, weight: "bold", fill: fit-blue)[Thank you for your time]
        #v(2em)
        #text(size: 24pt)[Questions?]
    ]
]

#slide("Opponent's questions")[
    #set text(size: 18pt)

    *Q1:* [Nothing, I'm so good]

    - [No assessment yet]

    #v(1em)

    *Q2:* [...]

    - [...]
]
