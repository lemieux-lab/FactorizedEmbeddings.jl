push!(LOAD_PATH,"../src/")
using FactorizedEmbeddings
using Documenter
makedocs(
         sitename = "FactorizedEmbeddings.jl",
         modules  = [FactorizedEmbeddings],
         pages=[
                "Home" => "index.md"
               ])
deploydocs(;
    repo="github.com/lemieux-lab/FactorizedEmbeddings.jl",
    # remotes="https://github.com/lemieux-lab/FactorizedEmbeddings.jl.git"
)