#include <cstdint>
#include <functional>

#include "dnnl_threadpool_iface.hpp"

#include "jlcxx/jlcxx.hpp"
#include "jlcxx/stl.hpp"

// Allow std::function to be called from Julia
class Opaque {
  public:
    Opaque(const std::function<void(int,int)>& fn) : fn(fn) {}
    const std::function<void(int,int)>& fn;
};

typedef Opaque* dnnl_callback;
typedef std::function<void(int,dnnl_callback)> julia_callback;


class threadpool : public dnnl::threadpool_interop::threadpool_iface {
  private:
    // Julia Registered Functions.
    std::function<bool()> _get_in_parallel;

    // Return an opaque function pointer.
    // Need to keep track on the Julia side to ensure that we call this function
    // correctly.
    julia_callback _parallel_for;
    int _num_threads;

  public:
    explicit threadpool(
        std::function<bool()> get_in_parallel,
        //std::function<void(int, const std::function<void(int, int)> &)>
        //    parallel_for,
        julia_callback parallel_for,
        int num_threads)
        : _get_in_parallel(get_in_parallel), _parallel_for(parallel_for),
          _num_threads(num_threads) {}

    int get_num_threads() const override { return _num_threads; }

    bool get_in_parallel() const override { return _get_in_parallel(); }

    uint64_t get_flags() const override { return 0; }

    void parallel_for(int n, const std::function<void(int, int)> &fn) override {
        auto ptr = std::make_unique<Opaque>(fn);
        _parallel_for(n, ptr.get());
    }
};

JLCXX_MODULE define_julia_module(jlcxx::Module &mod) {
    typedef bool (*in_parallel_type)();
    typedef void (*parallel_for_type)(int, dnnl_callback);

    mod.add_type<threadpool>("OneDNNThreadpool");

    mod.method("call_opaque", [](void* _opaque, int i, int n){
        auto opaque = static_cast<Opaque*>(_opaque);
        opaque->fn(i, n);
    });

    mod.method("construct_threadpool", [](
        // Use "void*" at the interface so we can pass reasonable Julia functions.
        void* _get_in_parallel,
        void* _parallel_for,
        int num_threads
    ){
        auto get_in_parallel = std::function<bool()>((in_parallel_type)_get_in_parallel);
        auto parallel_for = std::function<void(int,dnnl_callback)>(
            (parallel_for_type) _parallel_for
        );
        return threadpool(get_in_parallel, parallel_for, num_threads);
    });
}
