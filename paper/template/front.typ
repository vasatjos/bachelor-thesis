#let title-page(
    print,
    title: "",
    author: (
        name: "",
        email: "",
        url: "",
    ),
    submission-date: datetime.today(),
    bachelor: true,
    faculty: "",
    department: "",
    supervisor: "",
    font: "New Computer Modern",
) = {
    // render as a separate page
    // margins taken from "new" fit-ctu template

    show: page.with(margin: (top: 40mm, bottom: 40mm, left: 50mm, right: 35mm))

    // faculty logo
    image("res/logo-fit-en-black.svg", width: 200pt)
    v(25mm)

    set align(left)
    set place(left)
    set text(font: font)

    {
        if bachelor [
            Bachelor's Thesis
        ] else [
            Master's Thesis
        ]
    }

    v(10mm)

    {
        set text(size: 25pt, weight: "regular")
        [
            #upper(title)
        ]
    }

    v(5mm)
    {
        set text(size: 12.5pt, weight: "regular")
        [
            #author.name
        ]
    }

    v(3fr)
    {
        // set text(size: 10pt, weight: "regular")
        [
            Faculty of #faculty\
            Department of #department\
            Supervisor: #supervisor\
            \
            #submission-date.display("[day padding:none] [month repr:long] [year]")
        ]
    }
}

#let imprint-page(
    print,
    title: "",
    author: (
        name: "",
        email: "",
        url: "",
    ),
    submission-date: datetime.today(),
    bachelor: false,
    faculty: "",
    department: "",
    supervisor: "",
) = {
    show: page.with(margin: (bottom: 40mm, top: 46mm, left: 39.5mm, right: 40mm))
    //  show: page.with(margin: (bottom: 40mm, top:46mm, inside:47mm, outside:32.5mm))

    set text(weight: "extralight")
    set par(justify: true)
    set align(bottom)

    [
        Czech Technical University in Prague\
        Faculty of #faculty\
        #sym.copyright
        #submission-date.display("[year]")
        #author.name. All rights reserved. \
        #[
            #set text(style: "italic")
            This thesis is school work as defined by Copyright Act of the Czech Republic. It has been
            submitted at Czech Technical University in Prague, Faculty of Information Technology. The
            thesis is protected by the Copyright Act and its usage without author’s permission is prohibited
            (with exceptions defined by the Copyright Act).
        ]

        #v(1em)
        Citation of this thesis:
        #author.name.
        #{
            set text(style: "italic")
            title
        }.
        #if bachelor [
            Bachelor's Thesis.
        ] else [
            Master's Thesis.
        ]
        Czech Techincal University in Prague,
        Faculty of #faculty,
        #submission-date.display("[year]").
    ]
}


#let abstract-page(
    print,
    submission-date,
    abstract-en: [],
    abstract-cz: [],
    acknowledgement: [],
    declaration: [
        I declare that the presented work was developed independently and that I have listed all sources of information used within it in accordance with the methodical instructions for observing the ethical principles in the preparation of university theses.
    ],
) = {
    // render as a separate page; add room at the bottom for TODOs and notes
    // \if@twoside
    //   \RequirePackage[top=4.6cm,bottom=4cm,footskip=4em,inner=4.7cm,outer=3.25cm]{geometry}[2020/01/02] %page layout
    // \else
    //   \RequirePackage[left=3.95cm,right=4.0cm,top=4.6cm,bottom=4cm,footskip=4em]{geometry}[2020/01/02] %page layout
    // \fi

    //
    //
    if print {
        show: page.with(margin: (bottom: 40mm, top: 46mm, inside: 47mm, outside: 32.5mm))
    } else {
        show: page.with(margin: (bottom: 40mm, top: 46mm, left: 39.5mm, right: 40mm))
    }

    set heading(outlined: false, bookmarked: false)
    // pretty hacky way to disable the implicit linebreak in my heading style
    show heading: it => {
        show pagebreak: it => { linebreak() }
        block(it)
        //block(it, above: 2pt)
    }

    // no idea why there is a margin here
    v(-30.2pt)
    [
        = Abstract
        #abstract-en
    ]

    [
        = Abstrakt (CZ)
        #abstract-cz
    ]

    v(6.6pt)
    //v(-6pt)
    grid(
        columns: (47.5%, 47.5%),
        gutter: 5%,
        [
            = Acknowledgement
            #set text(style: "italic")
            #acknowledgement
        ],

        [
            = Declaration
            #declaration

            In Prague, #submission-date.display("[day]. [month]. [year]")

            #v(2em)
            #repeat[.]
        ],
    )

    context {
        set text(size: 15pt, weight: "bold")
        set align(center)

        v(1em)
        grid(
            columns: (47%, 47%),
            gutter: 6%,
            {
                let todo-count = counter("todo").final().at(0)
                if (todo-count > 0) {
                    set text(fill: red)
                    block(width: 100%, inset: 4pt)[#todo-count TODOs remaining]
                }
            },
            {
                let note-count = counter("note").final().at(0)
                if (note-count > 0) {
                    block(fill: yellow, width: 100%, inset: 4pt)[#note-count notes]
                }
            },
        )
    }
}
