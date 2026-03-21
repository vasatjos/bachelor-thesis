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
        I would like to express my gratitude to my dear friends and family for their support and encouragement
        during my whole academic journey.
        I would also like to thank my supervisor, Ing. Daniel Vašata, Ph.D.,
        for his help and insight during my time writing this thesis.
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

= Introduction

#Gls("rl") has shown superhuman performance in many board games,
with AlphaZero famously beating the best go players in the world. However, an area that remains
much less explored are games where a massive part of the world-state remains unknown
to the models.

Prší is a popular card game in the Czech Republic, it is one of the many
variants of the German game Mau-Mau, not dissimilar to Uno. Its rules will be explained
in a following chapter.
As a card game where everything but your hand and what has already been played
is a mistery to the player, it provides a suitable environment for exploring
#gls("rl") methods in a stochastic setting.


#lorem(120)

#lorem(40)

#lorem(70)

#heading([Goals], depth: 2, numbering: none, outlined: false)

= Background

== Part 1

#lorem(100) @template

=== Subpart 1

#lorem(40)

=== Subpart 2

#lorem(70)

== Part 2

#lorem(100)

= Future work

#lorem(100)

= Conclusion

#lorem(100)


#bibliography("bibliography.bib")


// all h1 headings from here on are appendices
#show: start-appendix

= Acronyms

#print-glossary(entry-list, show-all: false, user-print-glossary: acronym-table)

= An example appendix

#lorem(100)

$
sum_(i=1)^(infinity) 1 / i
$

= Code block

```cpp
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}
```
