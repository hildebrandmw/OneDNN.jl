# OneDNN

Pipeline Idea

Replace DNNL compatible ops with `Initializers`.
A first pass through the network will instantiate the `Initializers` into `Primitives`.
We can then walk the network functor and replace all the `Initializers` with their respective `Primitives`.

For training, we can run the `Initializer` network under Zygote and automatically switch the prop-kind and connect together forward and backward ops.

