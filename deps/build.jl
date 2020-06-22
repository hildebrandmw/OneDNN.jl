url = "https://github.com/oneapi-src/oneDNN/releases/download/v1.5/dnnl_lnx_1.5.0_cpu_iomp.tgz"
localpath = "dnnl.tgz"

download(url, localpath)
run(`tar -xvf $localpath`)
mv("dnnl_lnx_1.5.0_cpu_iomp", "dnnl")
