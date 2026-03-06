#import "./template/template.typ": *

#show: template.with(
  meta: (
    title: "An Interesting Thesis Title",
    author: (
      name: "Jan Novák",
    ),
    // submission-date: datetime(year: 2012, month: 1, day: 21),
    submission-date: datetime.today(),

    // true for bachelor's thesis, false for master's thesis
    bachelor: true,
    faculty: "Information Technology",
    department: "Lollygagging",
    supervisor: "Ing. Jan Novák, PhD.",
  ),

  // set to true if generating a PDF for print (shifts page layout, turns code blocks greyscale, correctly aligns odd/even pages,...)
  print: false,

  abstract-en: [
    #lorem(40)
    
    #lorem(60)
  ],

  abstract-cz: [
    #lorem(40)
    
    #lorem(60)
  ],

  acknowledgement: [
    #lorem(30)
    
    #lorem(30)
  ],
    keywords-en: [
    #lorem(10)
  ],
  keywords-cz: [
    #lorem(10)
  ],
  assignment: read("assignment.pdf", encoding: none)
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

= An example appendix

#lorem(100)
