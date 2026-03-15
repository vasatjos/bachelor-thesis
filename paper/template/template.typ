#import "./front.typ": *

#let template(
    meta: (),
    font: "Libertinus Serif",
    print: false,
    acknowledgement: "",
    declaration: "",
    abstract-en: "",
    abstract-cz: "",
    keywords-en: "",
    keywords-cz: "",
    assignment: "",
    ..intro-args,
    body,
) = {
    set document(
        author: meta.author.name,
        title: meta.title,
        date: meta.submission-date,
    )

    set text(font: font, size: 11pt, lang: "en", fallback: false)

    // the idea behind the inner margin is that if you lay the book out flat, there should be the same amount of space in the middle as on the outside
    // however, with binding from Haron.cz, the sides do not lay flat; from discussion with the guy at Haron, 8mm is consumed by the binding, and quite a lot of extra space is lost, since the page does not lay flat due to the binding; when I measured it in my printed thesis, I think the inner margin should be roughly 34mm and the outer margin 36mm at the beginning and end (where one side lays flat), and a bit more than that in the middle, but the variance based on how much you press down on the page is quite high, so we can make our lives easier and go with a symmetric 35mm+35mm margin
    let a4-width = 210mm
    // let text-width = 140mm // original value from template
    let text-width = 130mm
    let margin = (a4-width - text-width) / 2


    set page(
        paper: "a4",
        // same top/bottom margin as inner/outer; looks good in the PDF version
        margin: margin,
    )


    if assignment.len() == 0 {
        page[Replace this page with Assignment]
    }

    if print {
        page[]
    }

    // render title page before configuring the rest, which we don't use
    title-page(print, font: font, ..meta)

    pagebreak()

    if print {
        page[]
    }

    {
        hide[= Acknowledgements]
        set par(justify: true)
        set text(weight: "extralight", style: "italic")
        v(1fr)
        block(width: 60%, acknowledgement)
        v(2fr)

        if print {
            page[]
        }
    }
    pagebreak()

    {
        set par(justify: true)

        v(1fr)
        [
            #set align(right)
            = Declaration]
        declaration
        v(1.5em)

        [In Prague on ]
        meta.submission-date.display("[day]. [month]. [year]")
        h(1fr)
        box(width: 1fr, repeat[.])
    }

    pagebreak()

    imprint-page(print, ..meta)


    pagebreak()

    set page(numbering: "i")

    {
        v(28mm)
        [ = Abstract]
        abstract-en
        v(1em)
        [
            #set text(weight: "bold")
            Keywords:
        ]
        h(1em)
        keywords-en

        if print {
            pagebreak()
        }
        v(28mm)
        [ = Abstrakt]
        abstract-cz
        v(1em)
        [
            #set text(weight: "bold")
            Klíčová slova:
        ]
        h(1em)
        keywords-cz
    }
    pagebreak()

    outline(depth: 2, indent: auto)
    pagebreak()
    // [ = List of Figures]
    outline(title: "List of Figures", target: figure.where(kind: image))
    // [ = List of Tables]
    outline(title: "List of Tables", target: figure.where(kind: table))
    // [ = List of Code Listings]
    outline(title: "List of Code Listings", target: figure.where(kind: raw))


    set par(justify: true)
    set par(first-line-indent: 1.5em)

    set line(length: 100%, stroke: 1pt + luma(200))

    set figure(placement: auto)
    show figure.caption: set text(0.9em)
    show figure.caption: box.with(width: 92%)
    show figure.caption: par.with(justify: false)

    // Render code blocks with a grey background and external padding.
    show raw.where(block: true): it => {
        set par(justify: false)
        set align(left)
        v(8pt)
        block(
            width: 100%,
            fill: luma(248),
            spacing: 0pt,
            outset: 8pt,
            radius: 4pt,
        )[#it]
        v(8pt)
    }

    import "@preview/outrageous:0.3.0"
    set outline(indent: 1em)
    show outline.entry: outrageous.show-entry.with(
        font: (none, none),
        // very hacky way to format appendices differently
        // there's gotta be a better way, but I don't see it
        body-transform: (lvl, body) => {
            if "children" in body.fields() {
                let (num, ..text) = body.children
                if regex("^[A-Z]$") in num.text {
                    return "Appendix " + num + ": " + text.join()
                }
            }
            body
        },
    )

    set heading(supplement: "Chapter", numbering: "1.1")
    show heading.where(level: 1): it => {
        pagebreak(weak: true)

        show: block

        let use-supplement = it.outlined and it.numbering != none
        if (use-supplement) {
            text(size: 13pt, fill: rgb(120, 120, 120))[
                #it.supplement #counter(heading).display(it.numbering)
            ]
            linebreak()
            v(-16pt)
        }

        set align(end)
        text(size: 24pt, weight: "bold", font: font)[
            #it.body
        ]

        if (use-supplement) {
            v(22pt)
        } else {
            v(5.5pt)
        }
    }

    show heading.where(level: 2): it => {
        set text(size: 18pt, weight: "bold")
        block(it, below: 18pt, above: 32pt)
    }

    show heading.where(level: 3): it => {
        set text(size: 16pt, weight: "bold")
        block(it, below: 16pt, above: 22pt)
    }

    //   set bibliography(style: "chicago-notes", title: none)
    set bibliography(style: "res/IEEE-modified.csl", title: none)
    show bibliography: it => {
        heading("Bibliography")

        set text(size: 9pt)
        set par(justify: false)
        columns(2, it)
    }

    pagebreak(weak: true)

    // start numbering from the first page of actual text
    set page(numbering: "1")
    counter(heading).update(0)
    counter(page).update(1)

    body
}

// call this function after bibliography using an `everything show` rule:
//   #show: start-appendix
#let start-appendix(body) = {
    set heading(supplement: "Appendix", numbering: "A.1")
    counter(heading).update(0)
    body
}

// Renders a two-column borderless table of acronyms.
// `entries` is an array of (acronym, meaning) pairs, e.g.:
//   #acronym-table((("API", "Application Programming Interface"), ...))
#let acronym-table(entries) = {
    set par(first-line-indent: 0pt)
    table(
        columns: (auto, 1fr),
        stroke: none,
        inset: (x: 8pt, y: 5pt),
        align: (right, left),
        ..for (abbr, meaning) in entries {
            (
                text(weight: "bold")[#abbr],
                meaning,
            )
        }
    )
}

#let todo(msg) = {
    counter("todo").step()
    [#text(fill: red, weight: "bold")[TODO: #msg]]
}

#let note(msg) = {
    counter("note").step()
    [#block(fill: yellow, width: 100%, inset: 3pt, radius: 3pt)[NOTE: #msg]]
}
