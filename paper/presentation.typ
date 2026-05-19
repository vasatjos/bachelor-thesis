#let fit-blue = rgb("0064bb")
#let logo = "template/res/logo-fit-cz-blue.svg"

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
                    #text(size: 12pt, fill: gray, [Posilované učení pro karetní hru Prší])
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
                text(size: 12pt, fill: gray, [Josef Vašata -- Obhajoba bakalářské práce]),
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

// --- Titulní snímek ---
#page(header: none, footer: none)[
    #align(center + horizon)[
        #image(logo, width: 25%)
        #v(1em)
        #text(size: 34pt, weight: "bold", fill: fit-blue)[Posilované učení pro karetní hru Prší]
        #v(0.5em)
        #text(size: 24pt)[Josef Vašata]
        #linebreak()
        #text(size: 14pt, style: "italic")[Obhajoba bakalářské práce]
        #v(2em)
        #grid(
            columns: (1fr, 1fr),
            align(left)[
                #text(size: 16pt)[*Vedoucí:* Ing. Daniel Vašata, Ph.D.] \
                #text(size: 16pt)[*Obor:* Umělá inteligence]
            ],
            align(right)[
                #text(size: 16pt)[Květen 2026]
            ],
        )
    ]
]

// --- Snímky ---

#slide("Motivace")[
    - Prší -- populární hra v ČR
    - Zajímavý problém pro AI
        - Stochasticita
        - Neúplná informace
]

#slide("Cíle práce")[
    1. Implementace prostředí pro Prší
    2. Srovnání více RL přístupů:
        - Tabulkové metody: Monte Carlo, Q-Learning
        - Hluboké učení: DQN, REINFORCE
    3. Vyhodnocení:
        - Úspěšnost proti hladové heuristice
        - Úspěšnost proti lidským hráčům
]

#slide("Formalizace problému")[
    - Hra je modelována jako _částečně pozorovatelný Markovský rozhodovací proces_ (POMDP)
    // - *Prostor pozorování*:
    //     - Karty v ruce, horní karta + barva, efekt (např. "ber 2"), počet karet protihráče
    // - *Prostor akcí*:
    //     - Validní karty k vynesení, volba barvy (pro svrška)
    #align(center, image("images/pomdp.png", height: 40%))
]

#slide("Evaluované algoritmy")[
    - Tabulkové metody (MC, Q-Learning):
        - Metody založené na odhadu Q funkce
        - Vyžadují diskrétní stavový prostor
    - Deep Q-Network (DQN):
        - Metoda založená na Q funkci
        - Využívá neuronové sítě
        - Podobné stavy = podobná hodnota
    - REINFORCE:
        - Metoda *policy gradient* (Přímo optimalizuje strategii $pi_theta(a|s)$)
]

#slide("Výsledky trénování")[
    #grid(
        columns: (1.2fr, 1fr),
        gutter: 1em,
        [
            - *REINFORCE* dosáhl *65% úspěšnosti* proti hladové strategii
            - Tabulkové metody dosáhly přibližně 50 %
            - DQN divergovalo
        ],
        [
            #image("images/reinforce_training.svg", width: 100%)
            #align(center)[#text(size: 14pt)[_Průběh úspěšnosti REINFORCE během trénování_]]
        ],
    )
]

#slide("Výsledky testování proti lidem")[
    - Srovnání výkonu:
        - Člověk vs. hladový agent: ~65% úspěšnost člověka
        - Člověk vs. REINFORCE: *~54%* úspěšnost člověka
    - Agent REINFORCE výrazně snížil převahu člověka
]

#slide("Přínos práce")[
    - Implementace prostředí:
        - Rozšiřitelné prostředí pro Prší v Pythonu
        - Následuje API Gymnasium
    - Benchmarking:
        - Srovnání moderních a tradičních metod posilovaného učení ve stochastickém
          prostředí
    - Evaluace proti lidem:
        - Vytvoření terminálového rozhraní pro testování agentů proti reálným hráčům
        - Sběr a analýza dat z interaktivního testování
]

#slide("Závěr a budoucí práce")[
    - Shrnutí:
        - Metody policy gradient jsou pro Prší vhodnější než hodnotové metody
    - Budoucí práce:
        - Reprezentace stavu: Využití RNN pro lepší paměť
        - Pokročilé algoritmy: Proximal Policy Optimization, Soft Actor-Critic (SAC),
            MuZero
]

#slide("Děkuji za pozornost")[
    #align(center + horizon)[
        #text(size: 40pt, weight: "bold", fill: fit-blue)[Děkuji za pozornost!]
        #v(2em)
        #text(size: 24pt)[Dotazy?]
    ]
]

#slide("Otázky oponenta")[
    #set text(size: 18pt)
    /*
    Zde doplňte otázky z posudku oponenta a své odpovědi.
    Například:
    */

    *Otázka 1:* [Žádná není, jsem fakt dobrej]

    - [Nemám zatím posudek]

    #v(1em)

    *Otázka 2:* [...]

    - [...]
]
