@testset "examples" begin

function find_sources(path::String, sources=String[])
    if isdir(path)
        for entry in readdir(path)
            find_sources(joinpath(path, entry), sources)
        end
    elseif endswith(path, ".jl")
        push!(sources, path)
    end
    sources
end

examples_dir = realpath(joinpath(@__DIR__, "..", "examples"))
examples = find_sources(examples_dir)
filter!(file -> readline(file) != "# EXCLUDE FROM TESTING", examples)

@testset for example in examples
    cd(dirname(example)) do
        cmd = `$(Base.julia_cmd()) --project=$(Base.current_project()) $example`
        @test success(pipeline(cmd, stderr=stderr))
    end
end

end
