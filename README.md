# FactorizedEmbeddings.jl

### Usage
```
model = FactorizedEmbeddings.fit(data; params) ## returns model::Flux.Chain
inferred = FactorizedEmbeddings.transform(model, data; params) # return transformed data
embeddings = FactorizedEmbeddings.fit_transform(model, data; params) # return transformed data
```

### Advanced 
```
model = FactorizedEmbeddings.get_embeddings(model; params) ## returns Tuple(Matrix{Float32}, Matrix{Float32}) 
```


### Management of GPU
To run on GPU, import required library beforehand. 
Runs with NVIDIA by default.
