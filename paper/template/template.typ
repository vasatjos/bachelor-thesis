#import "./front.typ": *

#let in-outline = state("in-outline", false)

#let flex-caption(short, long) = context if in-outline.get() { short } else { long }

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

    // Don't show chapters before the introduction in the thesis contents
    set heading(outlined: false)

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
        set par(justify: true)
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


    // alongside flex-caption allows for short and long figure captions
    // must be before any call to `outline`
    show outline: it => {
        in-outline.update(true)
        it
        in-outline.update(false)
    }

    outline(depth: 2, indent: auto)
    pagebreak()
    outline(title: "List of Figures", target: figure.where(kind: image))
    outline(title: "List of Tables", target: figure.where(kind: table))
    outline(title: "List of Code Listings", target: figure.where(kind: raw))


    set heading(outlined: true)

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
        // Reset the figure counters at each new chapter
        counter(figure.where(kind: image)).update(0)
        counter(figure.where(kind: table)).update(0)
        counter(figure.where(kind: raw)).update(0)

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

    // Make images inherit the chapter number (e.g., 2.1)
    show figure.where(kind: image): set figure(numbering: n => {
        let h-counter = counter(heading).at(here())
        if h-counter.len() > 0 {
            // h-counter.at(0) is the current Chapter (Level 1) number
            numbering("1.1", h-counter.at(0), n)
        } else {
            // Fallback for images appearing before the first chapter
            numbering("1", n)
        }
    })

    body
}

// call this function after bibliography using an `everything show` rule:
//   #show: start-appendix
#let start-appendix(body) = {
    set heading(supplement: "Appendix", numbering: "A.1")
    counter(heading).update(0)
    body
}

#import "@preview/glossarium:0.5.10": count-refs

// Renders a two-column borderless table of acronyms formatted for glossarium.
// AI generated, hopefully doesn't break
#let acronym-table(
    entries,
    groups,
    user-print-reference: none,
    ..args,
) = {
    set par(first-line-indent: 0pt)

    // Prevent glossarium figures from floating and breaking the table,
    // while keeping them intact so their <labels> survive for cross-referencing.
    show figure.where(kind: "glossarium_entry"): set figure(placement: none)
    show figure.where(kind: "glossarium_entry"): set block(above: 0pt, below: 0pt)

    // Respect glossarium's filtering arguments
    let named = args.named()
    let show-all = named.at("show-all", default: false)
    let min-refs = named.at("minimum-refs", default: 1)

    let visible-entries = entries.filter(e => show-all or count-refs(e.at("key")) >= min-refs)

    // Filter out arguments like `group-heading-level` that `user-print-reference` doesn't accept
    let valid-ref-args = (
        "show-all",
        "disable-back-references",
        "deduplicate-back-references",
        "minimum-refs",
        "description-separator",
        "shorthands",
        "user-print-title",
        "user-print-description",
        "user-print-back-references",
    )
    let ref-args = (:)
    for (k, v) in named {
        if k in valid-ref-args {
            ref-args.insert(k, v)
        }
    }

    table(
        columns: (auto, 1fr),
        stroke: none,
        inset: (x: 8pt, y: 5pt),
        // Sort alphabetically by the "sort" key (or fallback to "key")
        ..for entry in visible-entries.sorted(key: e => e.at("sort", default: e.at("key"))) {
            // Fallbacks in case an entry is missing a short or long form
            let short-form = entry.at("short")
            if short-form == none { short-form = entry.at("key") }

            let meaning = entry.at("long")
            if meaning == none { meaning = entry.at("description") }
            if meaning == none { meaning = [] }

            // Extract glossarium's page-number linking function
            let back-refs = context {
                // Only print them if references are enabled and the term is actually used
                if not named.at("disable-back-references", default: false) and count-refs(entry.at("key")) > 0 {
                    let print-refs = named.at("user-print-back-references")
                    let dedup = named.at("deduplicate-back-references", default: false)

                    // Add some spacing and the references in gray brackets
                    h(0.5em)
                    text(fill: luma(120))[[#print-refs(entry, deduplicate: dedup)]]
                }
            }

            (
                // Column 1: The abbreviation, wrapped in glossarium's reference generator
                user-print-reference(
                    entry,
                    ..ref-args,
                    user-print-gloss: (e, ..opts) => align(right)[#text(weight: "bold")[#short-form]],
                ),
                // Column 2: The meaning + page references
                [#meaning#back-refs],
            )
        }
    )
}
