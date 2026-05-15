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
        text(size: 12pt, fill: gray, [Posilované učení pro karetní hru Prší]),
        image(logo, height: 1.5em)
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
        text(size: 12pt, fill: gray, counter(page).display())
      )
    }
  }
)

#set text(font: "DejaVu Sans", size: 22pt, lang: "cs")
#set list(indent: 1em)

#let slide(title, body) = {
  pagebreak(weak: true)
  v(0.5em)
  block(
    width: 100%,
    inset: (bottom: 0.5em),
    text(size: 32pt, weight: "bold", fill: fit-blue, title)
  )
  v(1em)
  body
}

// --- Titulní snímek ---
#page(header: none, footer: none)[
  #align(center + horizon)[
    #image(logo, width: 30%)
    #v(2em)
    #text(size: 36pt, weight: "bold", fill: fit-blue)[Posilované učení pro karetní hru Prší]
    #v(1em)
    #text(size: 26pt)[Josef Vašata]
    #linebreak()
    #text(size: 14pt, style: "italic")[Obhajoba bakalářské práce]
    #grid(
      columns: (1fr, 1fr),
      align(left)[
        #text(size: 16pt)[Vedoucí práce:\ Ing. Daniel Vašata, Ph.D.]
      ],
      align(right)[
        #text(size: 16pt)[Květen 2026]
      ]
    )
  ]
]

// --- Snímky ---

#slide("Úvod do hry Prší")[
  - *Prší* je populární česká karetní hra
  - Hlavní výzvy pro AI:
    - _Stochasticita_: Náhodné lízání karet a míchání balíčku
    - _Neúplná informace_: Karty protihráčů jsou skryté
    - _Dynamika_: Změna pravidel během hry (např. svršek mění barvu)
  - Hraje se s *32 německými kartami* (mariášky)
]

#slide("Cíle práce")[
  1. *Prostředí*: Implementace výkonného prostředí pro Prší v jazyce Python
  2. *Algoritmy*: Srovnání více přístupů:
    - Tabulkové: Monte Carlo, Q-Learning
    - Hluboké učení: DQN, REINFORCE
  3. *Vyhodnocení*:
    - Úspěšnost proti *hladové* (greedy) heuristice
    - Úspěšnost proti *lidským hráčům*
]

#slide("Formalizace problému")[
  - Hra je modelována jako *částečně pozorovatelný Markovův rozhodovací proces*
  - *Prostor pozorování*:
    - Vlastní ruka, horní karta, aktuální barva, počet trestných karet
  - *Prostor akcí*:
    - Validní karty k vynesení, volba barvy (pro svrška)
  #align(center, image("images/pomdp.png", height: 40%))
]

#slide("Evaluované algoritmy")[
  - *Tabulkové metody (MC & Q-Learning)*:
    - Vyžadují diskretizaci stavového prostoru
    - Fungují pro menší hry, ale Prší je komplexní
  - *Deep Q-Network (DQN)*:
    - Metoda založená na hodnotové funkci, využívá neuronové sítě
    - Problémy s konvergencí kvůli vysokému rozptylu
  - *REINFORCE*:
    - Metoda *policy gradient*
    - Přímo optimalizuje strategii $pi_theta(a|s)$
    - Nejsouspěšnější algoritmus v této studii
]

#slide("Výsledky trénování (Agent vs. Agent)")[
  #grid(
    columns: (1.2fr, 1fr),
    gutter: 1em,
    [
      - *REINFORCE* dosáhl *65% úspěšnosti* proti hladové strategii
      - Tabulkové metody dosáhly přibližně 50 %
      - Hluboké hodnotové metody (DQN) byly méně stabilní
    ],
    [
      #image("images/reinforce_training.svg", width: 100%)
      #align(center)[#text(size: 14pt)[_Průběh úspěšnosti REINFORCE během trénování_]]
    ]
  )
]

#slide("Výsledky testování proti lidem")[
  - *Testovací fáze*: 284 odehraných her proti lidem
  - *Srovnání výkonu*:
    - Člověk vs. *Hladový agent*: ~65% úspěšnost člověka
    - Člověk vs. *REINFORCE*: *~54%* úspěšnost člověka
  - *Závěr*:
    - Agent REINFORCE výrazně snížil převahu člověka
    - Prokázána schopnost učení ve vysoce stochastickém prostředí
]

#slide("Závěr a budoucí práce")[
  - *Shrnutí*:
    - Metody policy gradient jsou pro Prší vhodnější než hodnotové metody
    - Vlastní prostředí se ukázalo jako efektivní pro benchmarking
  - *Budoucí práce*:
    - *Self-Play*: Trénování proti vlastním verzím
    - *Reprezentace stavu*: Využití RNN/LSTM pro lepší paměť
    - *Pokročilé algoritmy*: PPO nebo Soft Actor-Critic (SAC)
]

#slide("Děkuji za pozornost")[
  #align(center + horizon)[
    #text(size: 40pt, weight: "bold", fill: fit-blue)[Děkuji za pozornost!]
    #v(2em)
    #text(size: 24pt)[Dotazy?]
  ]
]
