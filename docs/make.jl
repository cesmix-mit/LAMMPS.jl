pushfirst!(LOAD_PATH, joinpath(@__DIR__, "..")) # add LAMMPS to environment stack

using LAMMPS
using Documenter
using DocumenterCitations
using Literate

DocMeta.setdocmeta!(LAMMPS, :DocTestSetup, :(using LAMMPS); recursive=true)

bib = CitationBibliography(joinpath(@__DIR__, "citations.bib"))

##
# Generate examples
##

const EXAMPLES_DIR = joinpath(@__DIR__, "..", "examples")
const OUTPUT_DIR   = joinpath(@__DIR__, "src/generated")

examples = [
    "Basic SNAP" => "snap",
    "Fitting SNAP" => "fitting_snap",
]

for (_, name) in examples
    example_filepath = joinpath(EXAMPLES_DIR, string(name, ".jl"))
    Literate.markdown(example_filepath, OUTPUT_DIR, documenter=true)
end

examples = [title=>joinpath("generated", string(name, ".md")) for (title, name) in examples]

makedocs(;
    modules=[LAMMPS],
    authors="CESMIX-MIT",
    repo="https://github.com/cesmix-mit/LAMMPS.jl/blob/{commit}{path}#{line}",
    sitename="LAMMPS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cesmix-mit.github.io/LAMMPS.jl",
        assets=String[],
        mathengine = MathJax3(),
    ),
    pages = [
        "Home" => "index.md",
        "Examples" => examples,
        "API" => "api.md",
    ],
    plugins = [bib],
    doctest = true,
    linkcheck = true,
    strict = true,
)

deploydocs(;
    repo="github.com/cesmix-mit/LAMMPS.jl",
    devbranch = "main",
    push_preview = true,
)
