using LAMMPS
using Documenter

DocMeta.setdocmeta!(LAMMPS, :DocTestSetup, :(using LAMMPS); recursive=true)

makedocs(;
    modules=[LAMMPS],
    authors="CESMIX-MIT",
    repo="https://github.com/cesmix-mit/LAMMPS.jl/blob/{commit}{path}#{line}",
    sitename="LAMMPS.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://cesmix-mit.github.io/LAMMPS.jl",
        assets=String[],
    ),
    pages = [
        "Home" => "index.md",
        "API" => "api.md",
    ],
    doctest = true,
    linkcheck = true,
    strict = true,
)

deploydocs(;
    repo="github.com/cesmix-mit/LAMMPS.jl",
)
