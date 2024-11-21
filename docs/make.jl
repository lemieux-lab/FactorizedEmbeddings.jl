push!(LOAD_PATH,"../src/")
using FactorizedEmbeddings
using Documenter
makedocs(
         sitename = "FactorizedEmbeddings.jl",
         modules  = [FactorizedEmbeddings],
         repo="github.com/lemieux-lab/FactorizedEmbeddings.jl",
         pages=[
                "Home" => "index.md"
               ],
         format=Documenter.HTML(;
              prettyurls = get(ENV, "CI", "false") == "true"),
              # assets=["assets/style.css"]
         )
deploydocs(;
    repo="github.com/lemieux-lab/FactorizedEmbeddings.jl",
    # remotes="https://github.com/lemieux-lab/FactorizedEmbeddings.jl.git"
)