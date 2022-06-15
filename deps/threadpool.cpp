#include <cstdint>
#include <functional>
#include <utility>

#include "dnnl_threadpool_iface.hpp"

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

// Allow std::function to be called from Julia
using dnnl_kernel = const std::function<void(int,int)>;

// Type conversion chain for the win!
template<typename R, typename ...A>
std::function<R(A...)> make_function(void* f) {
    return std::function<R(A...)>(reinterpret_cast<R(*)(A...)>(f));
}

class threadpool : public dnnl::threadpool_interop::threadpool_iface {
  public:
    explicit threadpool(
        void* get_in_parallel,
        void* parallel_for,
        int num_threads
        ): m_get_in_parallel{make_function<bool>(get_in_parallel)}
         , m_parallel_for{make_function<void, int, dnnl_kernel&>(parallel_for)}
         , m_num_threads{num_threads} {}

    int get_num_threads() const override { return m_num_threads; }
    bool get_in_parallel() const override { return m_get_in_parallel(); }
    uint64_t get_flags() const override { return 0; }

    // Delegate the calling of the parallel functions to a Julia function.
    // This function will be provided by making an `@ccallable` function on the Julia side.
    void parallel_for(int n, dnnl_kernel& fn) override { m_parallel_for(n, fn); }

  private:
    // Julia Registered Functions.
    const std::function<bool()> m_get_in_parallel;
    const std::function<void(int,dnnl_kernel&)> m_parallel_for;
    const int m_num_threads;
};

JLCXX_MODULE define_julia_module(jlcxx::Module &mod) {
    // Wrap the `dnnl_kernel` type and call operator.
    mod.add_type<dnnl_kernel>("dnnl_kernel");
    mod.method("call", [](dnnl_kernel& fn, int n, int tid){ fn(n, tid); });

    mod.add_type<threadpool>("OneDNNThreadpool");
    mod.method("construct_threadpool", [](
        void* get_in_parallel, void* parallel_for, int num_threads
    ){
        return threadpool(get_in_parallel, parallel_for, num_threads);
    });
}
