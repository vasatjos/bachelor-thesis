#import "./template/template.typ": *


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
        Reinforcement learning, particularly deep variants, has dominated fully
        observable games like chess and go. This thesis attempts to apply this machine learning paradigm to Prší,
        an imperfect-information card game akin to Uno, where agents face limited observability of opponents' hands
        and therefore moves that can be taken against them.

        A Gym-like environment for 1v1 playing is implemented in Python, enabling self-play and both
        cross-agent and human-AI evaluation. Several models -- including Monte Carlo, Q-Learning, DQN, Recurrent DQN --
        are trained with the models based on neural networks achieving results capable of consistently defeating human players.
    ],

    abstract-cz: [
        #lorem(40)

        #lorem(60)
    ],

    keywords-en: [
        machine learning, reinforcement learning, card games, Q-Learning, DQN, neural networks
    ],
    keywords-cz: [
        strojové učení, posilované učení, karetní hry, Q-Learning, DQN, neuronové sítě
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

= Introduction

#lorem(80) @template

#lorem(120)

#lorem(140)

#lorem(40)

#lorem(70)

= Background

== Part 1

#lorem(100)

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

// TODO: use glossary package
#acronym-table((
    ("API", "Application Programming Interface"),
    ("CPU", "Central Processing Unit"),
    ("CSS", "Cascading Style Sheets"),
    ("GUI", "Graphical User Interface"),
    ("HTML", "HyperText Markup Language"),
    ("HTTP", "HyperText Transfer Protocol"),
    ("JSON", "JavaScript Object Notation"),
    ("OS", "Operating System"),
    ("REST", "Representational State Transfer"),
    ("URL", "Uniform Resource Locator"),
))

= An example appendix

#lorem(100)

= Code block

```cpp
#include <iostream>

int main() {
  std::cout << "Hello, World!" << std::endl;
  return 0;
}
```
